"""
Aggregate results from SVISE evaluation on synthetic noisy dataset.

Reads all per-chunk CSV files, combines them, re-simulates each chunk
from the physical equation coefficients, applies a divergence filter
(|omega_sim| > threshold → unstable), and reports RMSE statistics
for stable chunks only.

Works for both 5-min and 1-hour chunks via --results-dir and --chunk-size.

Usage:
    python aggregate_results_svise_synthetic.py --run-name run_SLURM_4102404
    python aggregate_results_svise_synthetic.py --run-name run_SLURM_4102405 --results-dir results_synthetic_noisy_1h --chunk-size 3600
"""
import os
import re
import math
import glob
import pandas as pd
import numpy as np
import json
import datetime
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import warnings

# ============================================================
# Configuration — keep in sync with SINDy aggregation script
# ============================================================
SIGMA = 0        # SVISE trains on raw (unsmoothed) data — sigma must be 0
DT    = 1.0

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4


# ============================================================
# Equation parsing (fast regex-based, no sympy)
# ============================================================

# Regex pattern to match a term like: -0.001051*omega**3, 1.2e-5*omega**2*theta, 0.005418
_NUM = r'[+-]?\s*\d+(?:\.\d*)?(?:e[+-]?\d+)?'
_TERM_PATTERN = re.compile(
    rf'({_NUM})\s*(?:\*\s*((?:omega|theta)(?:\s*\*\*\s*\d+)?(?:\s*\*\s*(?:omega|theta)(?:\s*\*\*\s*\d+)?)*))?',
    re.IGNORECASE
)

def _parse_var_powers(var_str):
    """Parse variable part like 'omega**2*theta' into {'omega': 2, 'theta': 1}."""
    powers = {'theta': 0, 'omega': 0}
    if not var_str:
        return powers
    # Split on * (but not **)
    parts = re.split(r'\*(?!\*)', var_str.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r'(theta|omega)(?:\s*\*\*\s*(\d+))?', part, re.IGNORECASE)
        if m:
            var = m.group(1).lower()
            exp = int(m.group(2)) if m.group(2) else 1
            powers[var] += exp
    return powers


def parse_physical_equation(eq_str):
    """
    Parse a physical equation string using regex (fast, no sympy).

    Returns list of coefficients: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    matching: c0 + c1*theta + c2*omega + c3*theta^2 + c4*theta*omega + c5*omega^2
              + c6*theta^3 + c7*theta^2*omega + c8*theta*omega^2 + c9*omega^3
    """
    if not isinstance(eq_str, str) or eq_str.strip().lower() in ("nan", "n/a", ""):
        return None
    if "FAILED" in eq_str or "Error" in eq_str:
        return None

    # Map (theta_power, omega_power) -> coefficient index
    power_to_idx = {
        (0, 0): 0,  # constant
        (1, 0): 1,  # theta
        (0, 1): 2,  # omega
        (2, 0): 3,  # theta^2
        (1, 1): 4,  # theta*omega
        (0, 2): 5,  # omega^2
        (3, 0): 6,  # theta^3
        (2, 1): 7,  # theta^2*omega
        (1, 2): 8,  # theta*omega^2
        (0, 3): 9,  # omega^3
    }

    coeffs = [0.0] * 10

    try:
        for m in _TERM_PATTERN.finditer(eq_str):
            coeff_str = m.group(1).replace(" ", "")
            if not coeff_str or coeff_str in ("+", "-"):
                continue
            coeff_val = float(coeff_str)
            var_str = m.group(2) if m.group(2) else ""
            powers = _parse_var_powers(var_str)
            key = (powers['theta'], powers['omega'])
            if key in power_to_idx:
                coeffs[power_to_idx[key]] += coeff_val
        return coeffs
    except Exception:
        return None


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


def prepare_synthetic_chunk(chunk, sigma=0):
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

def simulate_chunk(t_arr, theta0, omega0, coeffs):
    """Forward-simulate the ODE from polynomial coefficients.

    Uses mxstep=5000 to bail out quickly on stiff/divergent equations
    instead of grinding with tiny step sizes.
    """
    c = list(coeffs) + [0.0] * (10 - len(coeffs))

    def rhs(y, t):
        th, om = y
        dw = (c[0] + c[1]*th + c[2]*om + c[3]*th**2 + c[4]*th*om + c[5]*om**2
              + c[6]*th**3 + c[7]*th**2*om + c[8]*th*om**2 + c[9]*om**3)
        return [om, dw if math.isfinite(dw) else 0.0]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = odeint(rhs, [theta0, omega0], t_arr, full_output=False,
                         mxstep=5000)
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


def describe(series, name=""):
    """Compute descriptive stats for a numeric series, ignoring NaN."""
    s = series.dropna()
    if len(s) == 0:
        return {"count": 0}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SVISE synthetic noisy results with divergence filtering"
    )
    parser.add_argument("--run-name", type=str, required=True,
                        help="Folder name inside results dir, e.g. run_SLURM_4102404")
    parser.add_argument("--results-dir", type=str, default="results_synthetic_noisy",
                        help="Results parent folder (default: results_synthetic_noisy)")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Chunk size in samples (300 for 5-min, 3600 for 1-hour)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, args.results_dir, args.run_name)

    # Find all per-chunk CSV files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Expected column names (matching run_svise_synthetic_noisy.py output)
    expected_columns = [
        "Active_Chunk_Index", "Original_Chunk_Index",
        "Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
        "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
        "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
        "Eq_Theta", "Eq_Omega", "Eq_Omega_Physical"
    ]

    # Combine all CSVs
    dfs = []
    for f in csv_files:
        with open(f, 'r') as fh:
            first_line = fh.readline().strip()
        has_header = first_line.startswith("Active_Chunk_Index")
        if has_header:
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(f, header=None, names=expected_columns)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of re-runs)
    combined = combined.drop_duplicates(subset=["Active_Chunk_Index"], keep="last")
    combined = combined.sort_values("Active_Chunk_Index").reset_index(drop=True)

    print(f"Total unique chunks: {len(combined)}")

    # Convert numeric columns
    numeric_cols = ["Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
                    "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
                    "Final_Loss"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results (GP state estimation succeeded)
    valid = combined[combined["Orig_RMSE_Omega"].notna()].copy()
    failed = combined[combined["Orig_RMSE_Omega"].isna()]

    print(f"Successful (GP):  {len(valid)}")
    print(f"Failed:           {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # ----------------------------------------------------------
    # Load synthetic data for re-simulation
    # ----------------------------------------------------------
    data_path = os.path.join(script_dir, "synthetic_with_wiener.npz")
    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
        return

    all_chunks = load_synthetic_data(data_path, chunk_size=args.chunk_size)
    n_total_chunks = len(all_chunks)

    # ----------------------------------------------------------
    # Parse physical equations and re-simulate with divergence filter
    # ----------------------------------------------------------
    print(f"\nParsing physical equations and re-simulating with |ω| > {OMEGA_DIVERGENCE_THRESHOLD} filter...")

    rmse_values = []
    gp_rmse_values = []
    loss_values = []
    parsed_coefficients = []  # for ground truth comparison
    n_stable    = 0
    n_divergent = 0
    n_sim_fail  = 0
    n_parse_fail = 0

    for _, row in valid.iterrows():
        orig_idx = int(row.get("Original_Chunk_Index", row["Active_Chunk_Index"]))

        if orig_idx < 0 or orig_idx >= n_total_chunks:
            n_sim_fail += 1
            continue

        # Parse the physical equation to get coefficients
        eq_phys = row.get("Eq_Omega_Physical", "N/A")
        coeffs = parse_physical_equation(eq_phys)

        if coeffs is None:
            n_parse_fail += 1
            continue

        chunk = all_chunks[orig_idx]
        t_arr, theta, omega = prepare_synthetic_chunk(chunk, sigma=SIGMA)

        # Simulate from physical equation coefficients
        theta_sim, omega_sim = simulate_chunk(t_arr, theta[0], omega[0], coeffs)

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

        # Also track GP RMSE and loss for stable chunks
        gp_rmse = row["Orig_RMSE_Omega"]
        if pd.notna(gp_rmse):
            gp_rmse_values.append(float(gp_rmse))
        loss = row["Final_Loss"]
        if pd.notna(loss) and np.isfinite(loss):
            loss_values.append(float(loss))

        # Store coefficients for ground truth comparison
        parsed_coefficients.append(coeffs)

        n_processed = n_stable + n_divergent + n_sim_fail + n_parse_fail
        if n_processed % 5000 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_sim_fail} sim-fail, {n_parse_fail} parse-fail) ...")

    # ----------------------------------------------------------
    # Report results
    # ----------------------------------------------------------
    n_processed = n_stable + n_divergent + n_sim_fail + n_parse_fail
    rmse_arr = np.array(rmse_values)
    gp_rmse_arr = np.array(gp_rmse_values)
    loss_arr = np.array(loss_values)

    W = 70
    print(f"\n{'=' * W}")
    print(f"SVISE SYNTHETIC AGGREGATED RESULTS (divergence-filtered)")
    print(f"{'=' * W}")
    print(f"Results dir: {args.results_dir}/{args.run_name}")
    print(f"Smoothing: sigma={SIGMA}")
    print(f"Chunk size: {args.chunk_size} ({args.chunk_size/60:.0f} min)")
    print(f"Divergence threshold: |omega_sim| > {OMEGA_DIVERGENCE_THRESHOLD}")
    print()
    print(f"Total chunks in CSV: {len(combined)}")
    print(f"  GP succeeded:                      {len(valid)}")
    print(f"  GP failed:                         {len(failed)}")
    print(f"  Equation parse failed:             {n_parse_fail}")
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

    if len(gp_rmse_arr) > 0:
        print(f"\nGP State-Estimation RMSE (omega) — stable chunks only:")
        print(f"  Mean:   {np.mean(gp_rmse_arr):.6e}")
        print(f"  Median: {np.median(gp_rmse_arr):.6e}")

    if len(loss_arr) > 0:
        print(f"\nLoss (-ELBO) — stable chunks only:")
        print(f"  Mean:   {np.mean(loss_arr):.4f}")
        print(f"  Median: {np.median(loss_arr):.4f}")

    print(f"{'=' * W}")

    # ----------------------------------------------------------
    # Coefficient statistics (stable chunks only)
    # ----------------------------------------------------------
    coeff_names = [
        "Coeff_Const", "Coeff_Theta", "Coeff_Omega",
        "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
        "Coeff_Theta3", "Coeff_Theta2Omega", "Coeff_ThetaOmega2", "Coeff_Omega3",
    ]

    coeff_stats = {}
    if len(parsed_coefficients) > 0:
        coeff_arr = np.array(parsed_coefficients)
        for i, col_name in enumerate(coeff_names):
            if i < coeff_arr.shape[1]:
                vals = coeff_arr[:, i]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    coeff_stats[col_name] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "median": float(np.median(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "q25": float(np.percentile(vals, 25)),
                        "q75": float(np.percentile(vals, 75)),
                        "count": int(len(vals)),
                    }

    # Print coefficient table
    print(f"\n{'=' * W}")
    print(f"COEFFICIENT STATISTICS — STABLE CHUNKS ONLY (n={len(parsed_coefficients)})")
    print(f"{'=' * W}")
    print(f"{'Coefficient':<25} {'Mean':>15} {'Std':>15} {'Median':>15}")
    print(f"{'-' * W}")
    for col_name in coeff_names:
        if col_name in coeff_stats:
            s = coeff_stats[col_name]
            print(f"  {col_name:<23} {s['mean']:>+15.8e} {s['std']:>15.8e} {s['median']:>+15.8e}")

    # Ground truth comparison
    gt_path = os.path.join(script_dir, "ground_truth_params.json")
    ground_truth = None
    gt_comparison = None
    if os.path.exists(gt_path) and len(parsed_coefficients) > 0:
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"\nGround truth loaded from: {gt_path}")

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
        print(f"{'Coefficient':<25} {'SVISE Mean':>15} {'Ground Truth':>15} {'Rel. Error':>12}")
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
                    "svise_mean": rec_mean,
                    "svise_std": coeff_stats[col]["std"],
                    "svise_median": coeff_stats[col]["median"],
                    "ground_truth": gt_val,
                    "label": label,
                }

        # Also show nonlinear terms (should be ~0)
        for col in coeff_names:
            if col not in gt_map and col in coeff_stats:
                rec_mean = coeff_stats[col]["mean"]
                print(f"  {col:<23} {rec_mean:>+15.8e} {0.0:>+15.8e} {abs(rec_mean):.4e}")
                gt_comparison[col] = {
                    "svise_mean": rec_mean,
                    "svise_std": coeff_stats[col]["std"],
                    "svise_median": coeff_stats[col]["median"],
                    "ground_truth": 0.0,
                    "label": col,
                }

        print(f"{'=' * W}")

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------
    stats = {
        "total_chunks_csv": int(len(combined)),
        "gp_succeeded": int(len(valid)),
        "gp_failed": int(len(failed)),
        "equation_parse_failed": n_parse_fail,
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
    if len(gp_rmse_arr) > 0:
        stats["gp_rmse_omega_stable"] = {
            "mean": float(np.mean(gp_rmse_arr)),
            "std": float(np.std(gp_rmse_arr)),
            "median": float(np.median(gp_rmse_arr)),
        }
    if len(loss_arr) > 0:
        stats["loss_stable"] = {
            "mean": float(np.mean(loss_arr)),
            "std": float(np.std(loss_arr)),
            "median": float(np.median(loss_arr)),
        }
    stats["coefficient_stats_stable"] = coeff_stats
    if gt_comparison:
        stats["ground_truth_comparison"] = gt_comparison
    stats["config"] = {
        "sigma": SIGMA,
        "chunk_size": args.chunk_size,
        "dataset": "synthetic_with_wiener",
    }
    stats["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    combined_csv_path = os.path.join(results_dir, "all_chunks_combined.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"\nCombined CSV saved to: {combined_csv_path}")

    stats_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Aggregated summary saved to: {stats_path}")


if __name__ == "__main__":
    main()
