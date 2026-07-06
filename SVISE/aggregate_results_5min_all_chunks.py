"""
Aggregate results from all SVISE SLURM array job outputs.

Reads all per-chunk CSV files from results_5min_all_chunks/<run-name>/,
combines them, re-simulates each chunk from its stored equation string,
applies a divergence filter (|omega_sim| > threshold → unstable), and reports
RMSE statistics for stable chunks only.

Usage:
    python aggregate_results_5min_all_chunks.py --run-name run_SLURM_12345
"""
import os
import math
import glob
import pandas as pd
import numpy as np
import json
import datetime
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint

# ============================================================
# Configuration
# ============================================================
F_REF      = 60.0
DT         = 1.0
CHUNK_SIZE = 300
T_SCALE    = 30.0   # SVISE integrator time scaling

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4

# ELBO loss threshold: chunks with Final_Loss below this are considered
# unstable training runs and excluded from the analysis.
LOSS_MIN_THRESHOLD = -50000

# Expected column names in per-chunk CSVs
EXPECTED_COLUMNS = [
    "Chunk_Index", "Chunk_Start_Time",
    "Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
    "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
    "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
    "Eq_Theta", "Eq_Omega", "Eq_Omega_Physical"
]


# ============================================================
# Data loading (same as SVISE runner scripts)
# ============================================================

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path} ...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(
            method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(
            method='time', limit=limit_interpolation)
    return data


def get_all_valid_chunks(data):
    print("Finding all valid 5-minute chunks in the dataset ...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(
            subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))
    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == CHUNK_SIZE:
            valid_chunks.append((chunk_start, group))

    print(f"Found {len(valid_chunks)} valid 5-minute chunks.")
    return valid_chunks


def prepare_chunk(chunk_df, sigma=0, dt=DT):
    """Prepare theta and omega (no smoothing by default for SVISE)."""
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - F_REF) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    omega = gaussian_filter1d(omega_raw, sigma=sigma) if sigma > 0 else omega_raw.copy()
    theta = np.cumsum(omega) * dt
    t = np.arange(len(omega)) * dt
    return t, theta, omega


def compute_scaling_params(theta, omega, t_scale=T_SCALE):
    """Recompute the same scaling params used during SVISE training."""
    import torch
    train_x = torch.tensor(np.stack([theta, omega], axis=1), dtype=torch.float32)
    mean_x = train_x.mean(dim=0).numpy()
    std_x = train_x.std(dim=0).numpy()
    std_x[std_x < 1e-6] = 1.0
    mean_x[1] = 0.0
    std_x[0] = std_x[1] * t_scale
    return mean_x, std_x


# ============================================================
# Equation parsing & ODE simulation (from plot_forward_sim_vs_empirical.py)
# ============================================================

def parse_equation(eq_str):
    """Parse a polynomial equation string into coefficients dict."""
    coeffs = {
        "1": 0.0,
        "theta": 0.0, "omega": 0.0,
        "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0,
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }
    eq_str = eq_str.replace("+ -", "+-").replace("- ", "-").replace("  ", " ").strip()
    terms = []
    current_term = ""
    for char in eq_str:
        if char == '+' and current_term.strip():
            terms.append(current_term.strip())
            current_term = ""
        else:
            current_term += char
    if current_term.strip():
        terms.append(current_term.strip())

    term_patterns = [
        ("theta^3", "theta^3"), ("theta^2 omega", "theta^2 omega"),
        ("theta omega^2", "theta omega^2"), ("omega^3", "omega^3"),
        ("theta^2", "theta^2"), ("theta omega", "theta omega"),
        ("omega theta", "theta omega"), ("omega^2", "omega^2"),
        ("theta", "theta"), ("omega", "omega"),
    ]
    for term in terms:
        term = term.strip()
        if not term:
            continue
        matched = False
        for pattern, coeff_key in term_patterns:
            if pattern in term:
                coeff_str = term.replace(pattern, "").replace("*", "").strip()
                try:
                    coeffs[coeff_key] = float(coeff_str) if coeff_str else 1.0
                except ValueError:
                    pass
                matched = True
                break
        if not matched:
            try:
                coeffs["1"] = float(term)
            except ValueError:
                pass
    return coeffs


def simulate_ode(t, theta0, omega0, coeffs_omega, mean_x, std_x, t_scale=T_SCALE):
    """Simulate ODE in scaled space, unscale back."""
    x0 = np.array([theta0, omega0])
    x0_scaled = (x0 - mean_x) / std_x
    t_scaled = t / t_scale

    def drift(state, t_):
        th, om = state
        domega = (coeffs_omega["1"]
                  + coeffs_omega["theta"] * th + coeffs_omega["omega"] * om
                  + coeffs_omega["theta^2"] * th**2 + coeffs_omega["theta omega"] * th * om
                  + coeffs_omega["omega^2"] * om**2
                  + coeffs_omega["theta^3"] * th**3 + coeffs_omega["theta^2 omega"] * th**2 * om
                  + coeffs_omega["theta omega^2"] * th * om**2 + coeffs_omega["omega^3"] * om**3)
        if not math.isfinite(domega):
            domega = 0.0
        return [om, domega]

    sol_scaled = odeint(drift, x0_scaled, t_scaled, full_output=False)
    sol = sol_scaled * std_x + mean_x
    return sol[:, 0], sol[:, 1]  # theta_sim, omega_sim


def compute_rmse(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True,
                        help="Folder name inside results_5min_all_chunks, "
                             "e.g. run_SLURM_12345")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir,
                               "results_5min_all_chunks", args.run_name)

    # Find all per-chunk CSV files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # ----------------------------------------------------------
    # 1. Combine all per-chunk CSVs
    # ----------------------------------------------------------
    dfs = []
    for f in csv_files:
        with open(f, 'r') as fh:
            first_line = fh.readline().strip()
        has_header = first_line.startswith("Chunk_Index")
        if has_header:
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(f, header=None, names=EXPECTED_COLUMNS)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"Total unique chunks: {len(combined)}")

    # Convert numeric columns
    for col in ["Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
                "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
                "Final_Loss", "Diffusion_Theta", "Diffusion_Omega"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # ----------------------------------------------------------
    # 2. Load empirical data
    # ----------------------------------------------------------
    parquet_path = os.path.join(script_dir, "..", "dataset",
                                "South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(script_dir, "..", "dataset",
                               "Frequency_data_SK.pkl")
    if os.path.exists(parquet_path):
        data_path = parquet_path
    elif os.path.exists(pickle_path):
        data_path = pickle_path
    else:
        print(f"Error: Data file not found. Tried:\n  {parquet_path}\n  {pickle_path}")
        return

    data = load_data(data_path)
    all_chunks = get_all_valid_chunks(data)
    n_total_chunks = len(all_chunks)

    # ----------------------------------------------------------
    # 3. Re-simulate each chunk with divergence filter
    # ----------------------------------------------------------
    rmse_values  = []
    n_stable     = 0
    n_divergent  = 0
    n_sim_fail   = 0
    n_loss_fail  = 0   # ELBO loss exploded

    chunk_col = "Chunk_Index" if "Chunk_Index" in combined.columns else "Active_Chunk_Index"

    for _, row in combined.iterrows():
        chunk_id = int(row[chunk_col])
        eq_omega_str = str(row.get("Eq_Omega", ""))

        # Skip chunks with no valid equation
        if (not eq_omega_str or eq_omega_str == "nan"
                or "FAILED" in eq_omega_str
                or "nan" in eq_omega_str.lower()):
            n_sim_fail += 1
            continue

        # Skip chunks with exploded ELBO loss
        loss_val = row.get("Final_Loss", np.nan)
        if pd.notna(loss_val) and float(loss_val) < LOSS_MIN_THRESHOLD:
            n_loss_fail += 1
            continue

        if chunk_id < 0 or chunk_id >= n_total_chunks:
            n_sim_fail += 1
            continue

        chunk_start, chunk_df = all_chunks[chunk_id]
        t_arr, theta, omega = prepare_chunk(chunk_df, sigma=0)

        # Compute scaling params (same as SVISE training)
        mean_x, std_x = compute_scaling_params(theta, omega)

        # Parse equation and simulate in scaled space
        try:
            coeffs = parse_equation(eq_omega_str)
            theta_sim, omega_sim = simulate_ode(
                t_arr, theta[0], omega[0], coeffs, mean_x, std_x
            )
        except Exception:
            n_sim_fail += 1
            continue

        # Check for solver failure
        if np.all(np.isnan(omega_sim)):
            n_sim_fail += 1
            continue

        # Divergence check: |omega_sim| > threshold
        max_abs_omega = np.nanmax(np.abs(omega_sim))
        if max_abs_omega > OMEGA_DIVERGENCE_THRESHOLD:
            n_divergent += 1
            continue

        # Stable → compute RMSE against raw empirical omega
        rmse_omega = compute_rmse(omega_sim, omega)
        rmse_values.append(rmse_omega)
        n_stable += 1

        n_processed = n_stable + n_divergent + n_sim_fail + n_loss_fail
        if n_processed % 500 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_loss_fail} loss-fail, {n_sim_fail} sim-fail) ...")

    # ----------------------------------------------------------
    # 4. Report results
    # ----------------------------------------------------------
    n_processed = n_stable + n_divergent + n_sim_fail + n_loss_fail
    rmse_arr = np.array(rmse_values)

    print(f"\n{'=' * 60}")
    print(f"SVISE AGGREGATED RESULTS (divergence-filtered)")
    print(f"{'=' * 60}")
    print(f"Divergence threshold: |omega_sim| > {OMEGA_DIVERGENCE_THRESHOLD}")
    print(f"ELBO loss threshold:  loss < {LOSS_MIN_THRESHOLD}")
    print()
    print(f"Total chunks processed: {n_processed}")
    print(f"  Stable simulations:                {n_stable}")
    print(f"  Divergent/unstable (|ω|>{OMEGA_DIVERGENCE_THRESHOLD}):  {n_divergent}")
    print(f"  Unstable training (loss<{LOSS_MIN_THRESHOLD}): {n_loss_fail}")
    print(f"  Solver failures / no equation:     {n_sim_fail}")
    print()
    if len(rmse_arr) > 0:
        print(f"Forward-Simulated RMSE (omega) — stable chunks only:")
        print(f"  Mean:   {np.mean(rmse_arr):.6e}")
        print(f"  Std:    {np.std(rmse_arr):.6e}")
        print(f"  Median: {np.median(rmse_arr):.6e}")
        print(f"  Min:    {np.min(rmse_arr):.6e}")
        print(f"  Max:    {np.max(rmse_arr):.6e}")
        print(f"  25th %: {np.percentile(rmse_arr, 25):.6e}")
        print(f"  75th %: {np.percentile(rmse_arr, 75):.6e}")
    else:
        print("  No stable chunks found!")

    # GP state-estimation metrics (excluding loss-failed chunks)
    gp_valid = combined[combined["Orig_RMSE_Omega"].notna()].copy()
    if "Final_Loss" in gp_valid.columns:
        gp_valid = gp_valid[
            gp_valid["Final_Loss"].isna() |
            (gp_valid["Final_Loss"] >= LOSS_MIN_THRESHOLD)
        ]
    n_gp_valid = len(gp_valid)
    print()
    print(f"GP State-Estimation ({n_gp_valid} valid chunks, loss>={LOSS_MIN_THRESHOLD}):")
    if n_gp_valid > 0:
        print(f"  RMSE Omega:  mean={gp_valid['Orig_RMSE_Omega'].mean():.6e}  "
              f"median={gp_valid['Orig_RMSE_Omega'].median():.6e}")
        if "Orig_RMSE_Theta" in gp_valid.columns:
            print(f"  RMSE Theta:  mean={gp_valid['Orig_RMSE_Theta'].mean():.6e}  "
                  f"median={gp_valid['Orig_RMSE_Theta'].median():.6e}")
        if "Final_Loss" in gp_valid.columns:
            loss_valid = gp_valid["Final_Loss"].dropna()
            if len(loss_valid) > 0:
                print(f"  Loss:        mean={loss_valid.mean():.4f}  "
                      f"median={loss_valid.median():.4f}")
    print(f"{'=' * 60}")

    # Build stats dict
    stats = {
        "total_chunks": n_processed,
        "stable_chunks": n_stable,
        "divergent_chunks": n_divergent,
        "loss_fail_chunks": n_loss_fail,
        "sim_fail_chunks": n_sim_fail,
        "omega_divergence_threshold": OMEGA_DIVERGENCE_THRESHOLD,
        "loss_min_threshold": LOSS_MIN_THRESHOLD,
    }
    if len(rmse_arr) > 0:
        stats["sim_rmse_omega_stable"] = {
            "mean": float(np.mean(rmse_arr)),
            "std": float(np.std(rmse_arr)),
            "median": float(np.median(rmse_arr)),
            "min": float(np.min(rmse_arr)),
            "max": float(np.max(rmse_arr)),
            "q25": float(np.percentile(rmse_arr, 25)),
            "q75": float(np.percentile(rmse_arr, 75)),
        }
    stats["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # GP state-estimation stats (also excluding loss-failed chunks)
    gp_valid = combined[combined["Orig_RMSE_Omega"].notna()].copy()
    if "Final_Loss" in gp_valid.columns:
        gp_valid = gp_valid[
            gp_valid["Final_Loss"].isna() |
            (gp_valid["Final_Loss"] >= LOSS_MIN_THRESHOLD)
        ]
    if len(gp_valid) > 0:
        stats["gp_chunks_valid"] = int(len(gp_valid))
        stats["orig_rmse_omega"] = {
            "mean": float(gp_valid["Orig_RMSE_Omega"].mean()),
            "std": float(gp_valid["Orig_RMSE_Omega"].std()),
            "median": float(gp_valid["Orig_RMSE_Omega"].median()),
            "min": float(gp_valid["Orig_RMSE_Omega"].min()),
            "max": float(gp_valid["Orig_RMSE_Omega"].max()),
            "q25": float(gp_valid["Orig_RMSE_Omega"].quantile(0.25)),
            "q75": float(gp_valid["Orig_RMSE_Omega"].quantile(0.75)),
        }
        if "Orig_RMSE_Theta" in gp_valid.columns:
            stats["orig_rmse_theta"] = {
                "mean": float(gp_valid["Orig_RMSE_Theta"].mean()),
                "std": float(gp_valid["Orig_RMSE_Theta"].std()),
                "median": float(gp_valid["Orig_RMSE_Theta"].median()),
            }
        loss_valid = gp_valid["Final_Loss"].dropna() if "Final_Loss" in gp_valid.columns else pd.Series()
        if len(loss_valid) > 0:
            stats["loss"] = {
                "mean": float(loss_valid.mean()),
                "std": float(loss_valid.std()),
                "median": float(loss_valid.median()),
            }

    # Add diffusion stats if available
    if "Diffusion_Omega" in combined.columns:
        diff_omega_valid = combined["Diffusion_Omega"].dropna()
        diff_theta_valid = combined["Diffusion_Theta"].dropna() if "Diffusion_Theta" in combined.columns else pd.Series()
        if len(diff_omega_valid) > 0:
            stats["diffusion_omega"] = {
                "count": int(len(diff_omega_valid)),
                "mean": float(diff_omega_valid.mean()),
                "std": float(diff_omega_valid.std()),
                "median": float(diff_omega_valid.median()),
            }
        if len(diff_theta_valid) > 0:
            stats["diffusion_theta"] = {
                "count": int(len(diff_theta_valid)),
                "mean": float(diff_theta_valid.mean()),
                "std": float(diff_theta_valid.std()),
                "median": float(diff_theta_valid.median()),
            }

    # Save aggregated results
    combined_csv_path = os.path.join(results_dir, "all_chunks_combined.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"\nCombined CSV saved to: {combined_csv_path}")

    stats_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Aggregated summary saved to: {stats_path}")


if __name__ == "__main__":
    main()

