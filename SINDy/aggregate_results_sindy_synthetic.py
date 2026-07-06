"""
Aggregate results from SINDy SLURM array job on synthetic noisy data.

Reads all per-chunk CSV files from results_sindy_synthetic_all_chunks/<run-name>/,
combines them, re-simulates each chunk from its stored polynomial coefficients,
applies a divergence filter (|omega_sim| > threshold → unstable), and reports
RMSE statistics for stable chunks only.

Works for both 5-min and 1-hour chunks via --results-dir.

Usage:
    python aggregate_results_sindy_synthetic.py --run-name run_SLURM_4102407
    python aggregate_results_sindy_synthetic.py --run-name run_SLURM_XXX --results-dir results_sindy_synthetic_all_chunks_1h --chunk-size 3600
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
DT    = 1.0

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4


# ============================================================
# Data loading for synthetic data
# ============================================================

def load_synthetic_data(data_path, chunk_size=300):
    """Load synthetic data and chunk into windows."""
    print(f"Loading synthetic data from {data_path}...")

    with np.load(data_path) as data:
        omega = data['omega']
        theta = data['theta']

    print(f"  Total samples: {len(omega)}")

    n_chunks = len(omega) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append({
            'omega': omega[start:end],
            'theta': theta[start:end],
            'chunk_index': i,
        })

    print(f"  Total chunks (size={chunk_size}): {len(chunks)}")
    return chunks


def prepare_synthetic_chunk(chunk, sigma=15):
    """Prepare a synthetic chunk: apply Gaussian smoothing and compute theta."""
    omega_raw = chunk['omega'].copy()

    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
        theta = np.cumsum(omega) * DT
    else:
        omega = omega_raw.copy()
        theta = chunk['theta'].copy()

    t = np.arange(len(omega)) * DT
    return t, theta, omega


# ============================================================
# Forward simulation from polynomial coefficients
# ============================================================

def make_poly_rhs(coeffs, degree):
    """
    Build RHS for polynomial omega equation based on degree.

    Degree 1: d(omega)/dt = c0 + c1*θ + c2*ω
    Degree 2: d(omega)/dt = c0 + c1*θ + c2*ω + c3*θ² + c4*θω + c5*ω²
    Degree 3: d(omega)/dt = c0 + c1*θ + c2*ω + c3*θ² + c4*θω + c5*ω²
                            + c6*θ³ + c7*θ²ω + c8*θω² + c9*ω³
    """
    # Pad coefficients to 10 elements if needed
    c = list(coeffs) + [0.0] * (10 - len(coeffs))

    def rhs(t, y):
        theta, omega = y
        dw = c[0] + c[1] * theta + c[2] * omega
        if degree >= 2:
            dw += c[3] * theta**2 + c[4] * theta * omega + c[5] * omega**2
        if degree >= 3:
            dw += (c[6] * theta**3 + c[7] * theta**2 * omega
                   + c[8] * theta * omega**2 + c[9] * omega**3)
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
                        help="Folder name inside results dir")
    parser.add_argument("--results-dir", type=str,
                        default="results_sindy_synthetic_all_chunks",
                        help="Results parent folder (default: results_sindy_synthetic_all_chunks)")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Chunk size in samples (300 for 5-min, 3600 for 1-hour)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir, args.run_name)

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
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Could not read {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"Total unique chunks: {len(combined)}")

    # Detect coefficient columns dynamically
    coeff_cols = [c for c in combined.columns if c.startswith("Coeff_")]
    print(f"Detected {len(coeff_cols)} coefficient columns: {coeff_cols}")

    # Detect degree from number of coefficients
    if len(coeff_cols) <= 3:
        degree = 1
    elif len(coeff_cols) <= 6:
        degree = 2
    else:
        degree = 3
    print(f"Inferred polynomial degree: {degree}")

    # Convert numeric columns
    numeric_cols = ["Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total"] + coeff_cols
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Fill NaN coefficients with 0
    for col in coeff_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0.0)

    # ----------------------------------------------------------
    # 2. Load synthetic data
    # ----------------------------------------------------------
    data_path = os.path.join(script_dir, "..", "SVISE",
                             "synthetic_dataset_validation",
                             "synthetic_with_wiener.npz")
    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
        return

    all_chunks = load_synthetic_data(data_path, chunk_size=args.chunk_size)
    n_total_chunks = len(all_chunks)

    # ----------------------------------------------------------
    # 3. Re-simulate each chunk with divergence filter
    # ----------------------------------------------------------
    rmse_values = []
    stable_indices = []   # row indices of stable chunks (for coefficient analysis)
    n_stable    = 0
    n_divergent = 0
    n_sim_fail  = 0

    for _, row in combined.iterrows():
        chunk_id = int(row["Chunk_Index"])
        # Map to original chunk index if available
        orig_idx = int(row.get("Original_Chunk_Index", chunk_id))

        if orig_idx < 0 or orig_idx >= n_total_chunks:
            n_sim_fail += 1
            continue

        chunk = all_chunks[orig_idx]
        t_arr, theta, omega = prepare_synthetic_chunk(chunk, sigma=SIGMA)

        # Extract polynomial coefficients
        coeffs = [float(row[c]) for c in coeff_cols]

        # Build RHS and simulate
        rhs_func = make_poly_rhs(coeffs, degree)
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
        stable_indices.append(row.name)  # DataFrame row index
        n_stable += 1

        n_processed = n_stable + n_divergent + n_sim_fail
        if n_processed % 5000 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_sim_fail} sim-fail) ...")

    # ----------------------------------------------------------
    # 4. Report results
    # ----------------------------------------------------------
    n_processed = n_stable + n_divergent + n_sim_fail
    rmse_arr = np.array(rmse_values)

    print(f"\n{'=' * 60}")
    print(f"SINDy SYNTHETIC AGGREGATED RESULTS (divergence-filtered)")
    print(f"{'=' * 60}")
    print(f"Results dir: {args.results_dir}/{args.run_name}")
    print(f"Smoothing: sigma={SIGMA}")
    print(f"Polynomial degree: {degree}")
    print(f"Chunk size: {args.chunk_size} ({args.chunk_size/60:.0f} min)")
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
        "degree": degree,
        "chunk_size": args.chunk_size,
        "features": ["theta", "omega"],
        "dataset": "synthetic_with_wiener",
    }

    # ----------------------------------------------------------
    # 5. Coefficient statistics (stable chunks only)
    # ----------------------------------------------------------
    stable_df = combined.loc[stable_indices]

    # Ground truth
    gt_path = os.path.join(script_dir, "..", "SVISE",
                           "synthetic_dataset_validation",
                           "ground_truth_params.json")
    ground_truth = None
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"\nGround truth loaded from: {gt_path}")

    # Compute stats for each coefficient column
    coeff_stats = {}
    for col in coeff_cols:
        vals = pd.to_numeric(stable_df[col], errors="coerce").dropna()
        if len(vals) > 0:
            coeff_stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "count": int(len(vals)),
            }

    stats["coefficient_stats_stable"] = coeff_stats

    # Print coefficient table
    W = 75
    print(f"\n{'=' * W}")
    print(f"COEFFICIENT STATISTICS — STABLE CHUNKS ONLY (n={len(stable_df)})")
    print(f"{'=' * W}")
    print(f"{'Coefficient':<25} {'Mean':>15} {'Std':>15} {'Median':>15}")
    print(f"{'-' * W}")
    for col in coeff_cols:
        if col in coeff_stats:
            s = coeff_stats[col]
            print(f"  {col:<23} {s['mean']:>+15.8e} {s['std']:>15.8e} {s['median']:>+15.8e}")

    # Ground truth comparison
    if ground_truth:
        gt_c1 = ground_truth.get("c_1", float('nan'))
        gt_c2 = ground_truth.get("c_2", float('nan'))
        gt_dp = ground_truth.get("Delta_P", float('nan'))

        # Map: Coeff_Omega -> c_1, Coeff_Theta -> c_2, Coeff_Const -> Delta_P
        gt_map = {
            "Coeff_Omega": ("c_1 (damping)", gt_c1),
            "Coeff_Theta": ("c_2 (coupling)", gt_c2),
            "Coeff_Const": ("Delta_P (forcing)", gt_dp),
        }

        print(f"\n{'=' * W}")
        print(f"GROUND TRUTH COMPARISON — STABLE CHUNKS ONLY")
        print(f"{'=' * W}")
        print(f"{'Coefficient':<25} {'SINDy Mean':>15} {'Ground Truth':>15} {'Rel. Error':>12}")
        print(f"{'-' * W}")

        gt_comparison = {}
        for col, (label, gt_val) in gt_map.items():
            if col in coeff_stats:
                rec_mean = coeff_stats[col]["mean"]
                if gt_val != 0 and not np.isnan(gt_val):
                    rel_err = abs(rec_mean - gt_val) / abs(gt_val) * 100
                    rel_str = f"{rel_err:.1f}%"
                else:
                    rel_str = "N/A"
                print(f"  {label:<23} {rec_mean:>+15.8e} {gt_val:>+15.8e} {rel_str:>12}")
                gt_comparison[col] = {
                    "sindy_mean": rec_mean,
                    "sindy_std": coeff_stats[col]["std"],
                    "sindy_median": coeff_stats[col]["median"],
                    "ground_truth": gt_val,
                    "label": label,
                }

        # Also show nonlinear terms (should be ~0)
        for col in coeff_cols:
            if col not in gt_map and col in coeff_stats:
                rec_mean = coeff_stats[col]["mean"]
                print(f"  {col:<23} {rec_mean:>+15.8e} {0.0:>+15.8e} {abs(rec_mean):.4e}")
                gt_comparison[col] = {
                    "sindy_mean": rec_mean,
                    "sindy_std": coeff_stats[col]["std"],
                    "sindy_median": coeff_stats[col]["median"],
                    "ground_truth": 0.0,
                    "label": col,
                }

        print(f"{'=' * W}")
        stats["ground_truth_comparison"] = gt_comparison

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
