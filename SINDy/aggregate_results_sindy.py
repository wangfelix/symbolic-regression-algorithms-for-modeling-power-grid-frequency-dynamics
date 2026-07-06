"""
Aggregate results from SINDy SLURM array job outputs.

Reads all per-chunk CSV files from results_sindy_5min_all_chunks/<run-name>/,
combines them, re-simulates each chunk from its stored polynomial coefficients,
applies a divergence filter (|omega_sim| > threshold → unstable), and reports
RMSE statistics for stable chunks only.

Usage:
    python aggregate_results_sindy.py --run-name run_SLURM_12345_sindy
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
SIGMA = 15
F_REF = 60.0
DT    = 1.0
CHUNK_SIZE = 300

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4

# Coefficient column names in the per-chunk CSVs
# Degree-3 polynomial: [1, θ, ω, θ², θω, ω², θ³, θ²ω, θω², ω³]
COEFF_COLS = [
    "Coeff_Const", "Coeff_Theta", "Coeff_Omega",
    "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
    "Coeff_Theta3", "Coeff_Theta2Omega", "Coeff_ThetaOmega2", "Coeff_Omega3",
]


# ============================================================
# Data loading (same as run_sindy_5min_all_chunks.py)
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


def prepare_chunk(chunk_df, sigma=SIGMA):
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - F_REF) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    omega = gaussian_filter1d(omega_raw, sigma=sigma) if sigma > 0 else omega_raw.copy()
    theta = np.cumsum(omega) * DT
    t = np.arange(len(omega)) * DT
    return t, theta, omega


# ============================================================
# Forward simulation from polynomial coefficients
# ============================================================

def make_poly3_rhs(coeffs):
    """
    Build RHS for the full degree-3 polynomial omega equation.

    d(omega)/dt = c0 + c1*θ + c2*ω + c3*θ² + c4*θω + c5*ω²
                  + c6*θ³ + c7*θ²ω + c8*θω² + c9*ω³
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs

    def rhs(t, y):
        theta, omega = y
        dw = (c0
              + c1 * theta
              + c2 * omega
              + c3 * theta**2
              + c4 * theta * omega
              + c5 * omega**2
              + c6 * theta**3
              + c7 * theta**2 * omega
              + c8 * theta * omega**2
              + c9 * omega**3)
        return [omega, dw if math.isfinite(dw) else 0.0]
    return rhs


def simulate_chunk(t_arr, theta0, omega0, rhs_func):
    """Forward-simulate the ODE using odeint and return (theta_sim, omega_sim)."""
    def rhs_odeint(y, t):
        return rhs_func(t, y)

    try:
        sol = odeint(rhs_odeint, [theta0, omega0], t_arr, full_output=False)
        if sol.shape[0] == len(t_arr):
            return sol[:, 0], sol[:, 1]
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)


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
                        help="Folder name inside results_sindy_5min_all_chunks")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir,
                               "results_sindy_5min_all_chunks", args.run_name)

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
            df = pd.read_csv(f, header=None, names=[
                "Chunk_Index", "Chunk_Start_Time",
                "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
                "Eq_Theta", "Eq_Omega",
            ] + COEFF_COLS)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"Total unique chunks: {len(combined)}")

    # Convert numeric columns
    numeric_cols = ["Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total"] + COEFF_COLS
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Fill NaN coefficients with 0
    for col in COEFF_COLS:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0.0)

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
    rmse_values = []
    n_stable    = 0
    n_divergent = 0
    n_sim_fail  = 0

    for _, row in combined.iterrows():
        chunk_id = int(row["Chunk_Index"])

        if chunk_id < 0 or chunk_id >= n_total_chunks:
            n_sim_fail += 1
            continue

        chunk_start, chunk_df = all_chunks[chunk_id]
        t_arr, theta, omega = prepare_chunk(chunk_df, sigma=SIGMA)

        # Extract all 10 polynomial coefficients
        coeffs = [float(row[c]) for c in COEFF_COLS]

        # Build RHS and simulate
        rhs_func = make_poly3_rhs(coeffs)
        theta_sim, omega_sim = simulate_chunk(
            t_arr, theta[0], omega[0], rhs_func
        )

        # Check for solver failure
        if np.all(np.isnan(omega_sim)):
            n_sim_fail += 1
            continue

        # Divergence check: |omega_sim| > threshold
        max_abs_omega = np.nanmax(np.abs(omega_sim))
        if max_abs_omega > OMEGA_DIVERGENCE_THRESHOLD:
            n_divergent += 1
            continue

        # Stable → compute RMSE
        rmse_omega = compute_rmse(omega_sim, omega)
        rmse_values.append(rmse_omega)
        n_stable += 1

        n_processed = n_stable + n_divergent + n_sim_fail
        if n_processed % 500 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_sim_fail} sim-fail) ...")

    # ----------------------------------------------------------
    # 4. Report results
    # ----------------------------------------------------------
    n_processed = n_stable + n_divergent + n_sim_fail
    rmse_arr = np.array(rmse_values)

    print(f"\n{'=' * 60}")
    print(f"SINDy AGGREGATED RESULTS (divergence-filtered)")
    print(f"{'=' * 60}")
    print(f"Smoothing: sigma={SIGMA}")
    print(f"Divergence threshold: |omega_sim| > {OMEGA_DIVERGENCE_THRESHOLD}")
    print()
    print(f"Total chunks processed: {n_processed}")
    print(f"  Stable simulations:                {n_stable}")
    print(f"  Divergent/unstable (|ω|>{OMEGA_DIVERGENCE_THRESHOLD}):  {n_divergent}")
    print(f"  Solver failures (NaN):             {n_sim_fail}")
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
    print(f"{'=' * 60}")

    # Build stats dict
    stats = {
        "total_chunks": n_processed,
        "stable_chunks": n_stable,
        "divergent_chunks": n_divergent,
        "sim_fail_chunks": n_sim_fail,
        "omega_divergence_threshold": OMEGA_DIVERGENCE_THRESHOLD,
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
    stats["config"] = {
        "sigma": SIGMA,
        "degree": 3,
        "features": ["theta", "omega"],
    }
    stats["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save combined CSV and summary
    combined_csv_path = os.path.join(results_dir, "all_chunks_combined.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"\nCombined CSV saved to: {combined_csv_path}")

    stats_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Aggregated summary saved to: {stats_path}")


if __name__ == "__main__":
    main()
