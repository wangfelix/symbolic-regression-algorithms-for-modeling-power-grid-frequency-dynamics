"""
Aggregate results from SVISE evaluation on synthetic noiseless dataset (1-hour chunks).

Reads all per-chunk CSV files from results_synthetic_noiseless_1h/<run-name>/,
combines them, computes overall statistics, parses physical equations to
extract coefficients, and compares recovered coefficients to ground truth.

Usage:
    python aggregate_results_synthetic_1h.py --run-name run_SLURM_XXXXXXX
"""
import os
import re
import glob
import pandas as pd
import numpy as np
import json
import datetime
import argparse


def parse_physical_equation(eq_str):
    """
    Parse a physical equation string like:
        '-0.008454*omega - 1.8e-5*theta + 0.005418'
        '4.5e-5 - 0.002146*omega'
    Returns dict with keys: c_omega, c_theta, intercept (any may be None/0).
    """
    if not isinstance(eq_str, str) or eq_str.strip().lower() == "nan":
        return None

    c_omega = 0.0
    c_theta = 0.0
    intercept = 0.0

    # Match omega coefficient: e.g. -0.008454*omega or +0.008454*omega
    m_omega = re.search(r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*omega', eq_str, re.IGNORECASE)
    if m_omega:
        c_omega = float(m_omega.group(1).replace(" ", ""))

    # Match theta coefficient: e.g. -1.8e-5*theta or +0.00001*theta
    m_theta = re.search(r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*theta', eq_str, re.IGNORECASE)
    if m_theta:
        c_theta = float(m_theta.group(1).replace(" ", ""))

    # Intercept: any standalone number not attached to *omega or *theta
    # Remove the omega and theta terms, then parse what remains
    remainder = eq_str
    if m_omega:
        remainder = remainder[:m_omega.start()] + remainder[m_omega.end():]
    if m_theta:
        # Recalculate positions after first removal
        m_theta2 = re.search(r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*theta', remainder, re.IGNORECASE)
        if m_theta2:
            remainder = remainder[:m_theta2.start()] + remainder[m_theta2.end():]

    # Clean up remaining operators and whitespace
    remainder = remainder.strip().strip("+-").strip()
    if remainder:
        # Find any remaining number
        m_const = re.search(r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)', remainder)
        if m_const:
            try:
                intercept = float(m_const.group(1).replace(" ", ""))
            except ValueError:
                pass
            # Check if there was a leading minus before the omega/theta terms were removed
            # Re-parse from original string for sign correctness

    # More robust intercept: find all numeric tokens not followed by *omega or *theta
    all_terms = re.findall(r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)(?:\s*\*\s*(omega|theta))?', eq_str)
    intercept = 0.0
    for val_str, var_name in all_terms:
        val_str = val_str.replace(" ", "")
        if not val_str:
            continue
        if var_name == "":
            try:
                intercept += float(val_str)
            except ValueError:
                pass

    return {"c_omega": c_omega, "c_theta": c_theta, "intercept": intercept}


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


def main():
    parser = argparse.ArgumentParser(description="Aggregate synthetic noiseless results")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Folder name inside results_synthetic_noiseless_1h, e.g. run_SLURM_XXXXXXX")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results_synthetic_noiseless_1h", args.run_name)

    # Find all per-chunk CSV files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    # Expected column names (matching run_svise_synthetic_noiseless.py output)
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
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of re-runs)
    combined = combined.drop_duplicates(subset=["Active_Chunk_Index"], keep="last")
    combined = combined.sort_values("Active_Chunk_Index").reset_index(drop=True)

    print(f"\nTotal unique chunks: {len(combined)}")

    # Convert numeric columns
    numeric_cols = ["Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
                    "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
                    "Final_Loss"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results (GP state estimation succeeded)
    valid = combined[combined["Orig_RMSE_Omega"].notna()]
    failed = combined[combined["Orig_RMSE_Omega"].isna()]
    n_sim_valid = int(valid["Sim_RMSE_Omega"].notna().sum())

    print(f"Successful (GP):  {len(valid)}")
    print(f"Successful (Sim): {n_sim_valid}")
    print(f"Failed:           {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # =================================================================
    # Parse physical equations to extract coefficients
    # =================================================================
    parsed = valid["Eq_Omega_Physical"].apply(parse_physical_equation)
    valid = valid.copy()
    valid["Parsed_c_omega"] = parsed.apply(lambda x: x["c_omega"] if x else np.nan)
    valid["Parsed_c_theta"] = parsed.apply(lambda x: x["c_theta"] if x else np.nan)
    valid["Parsed_intercept"] = parsed.apply(lambda x: x["intercept"] if x else np.nan)

    n_parsed = int(valid["Parsed_c_omega"].notna().sum())
    print(f"Equations parsed: {n_parsed}")

    # =================================================================
    # Load ground truth parameters
    # =================================================================
    gt_path = os.path.join(script_dir, "ground_truth_params.json")
    ground_truth = None
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"Ground truth loaded from: {gt_path}")
    else:
        print(f"Warning: Ground truth not found at {gt_path}")

    # =================================================================
    # Compute statistics
    # =================================================================
    # Filter finite losses for loss stats (some chunks produce inf/-inf loss)
    finite_loss = valid["Final_Loss"][np.isfinite(valid["Final_Loss"])]

    stats = {
        "run_name": args.run_name,
        "total_chunks": int(len(combined)),
        "successful_chunks_gp": int(len(valid)),
        "successful_chunks_sim": n_sim_valid,
        "successful_chunks_parsed": n_parsed,
        "failed_chunks": int(len(failed)),
        "chunks_with_finite_loss": int(len(finite_loss)),
        "success_rate_gp": float(len(valid) / len(combined)),
        "success_rate_sim": float(n_sim_valid / len(combined)),
        "orig_rmse_omega": describe(valid["Orig_RMSE_Omega"]),
        "sim_rmse_omega": describe(valid["Sim_RMSE_Omega"]),
        "orig_rmse_theta": describe(valid["Orig_RMSE_Theta"]),
        "sim_rmse_theta": describe(valid["Sim_RMSE_Theta"]),
        "orig_rmse_total": describe(valid["Orig_RMSE_Total"]),
        "loss": describe(finite_loss),
        "coefficients": {
            "c_omega": describe(valid["Parsed_c_omega"]),
            "c_theta": describe(valid["Parsed_c_theta"]),
            "intercept": describe(valid["Parsed_intercept"]),
        },
    }

    # =================================================================
    # Ground Truth Comparison
    # =================================================================
    if ground_truth:
        gt_c1 = ground_truth.get("c_1", float('nan'))
        gt_c2 = ground_truth.get("c_2", float('nan'))
        gt_dp = ground_truth.get("Delta_P", float('nan'))
        gt_eps = ground_truth.get("epsilon", float('nan'))

        c_omega_valid = valid["Parsed_c_omega"].dropna()
        c_theta_valid = valid["Parsed_c_theta"].dropna()
        intercept_valid = valid["Parsed_intercept"].dropna()

        comparison = {
            "ground_truth": {
                "c_1 (c_omega)": gt_c1,
                "c_2 (c_theta)": gt_c2,
                "Delta_P (intercept sign varies)": gt_dp,
                "epsilon (noiseless=0)": gt_eps,
            },
            "recovered": {
                "c_omega": {
                    "mean": float(c_omega_valid.mean()) if len(c_omega_valid) > 0 else None,
                    "std": float(c_omega_valid.std()) if len(c_omega_valid) > 0 else None,
                    "median": float(c_omega_valid.median()) if len(c_omega_valid) > 0 else None,
                },
                "c_theta": {
                    "mean": float(c_theta_valid.mean()) if len(c_theta_valid) > 0 else None,
                    "std": float(c_theta_valid.std()) if len(c_theta_valid) > 0 else None,
                    "median": float(c_theta_valid.median()) if len(c_theta_valid) > 0 else None,
                },
                "intercept (|mean| ~ Delta_P)": {
                    "mean": float(intercept_valid.mean()) if len(intercept_valid) > 0 else None,
                    "abs_mean": float(intercept_valid.abs().mean()) if len(intercept_valid) > 0 else None,
                    "std": float(intercept_valid.std()) if len(intercept_valid) > 0 else None,
                },
            },
            "relative_errors": {},
        }

        # Relative errors
        if len(c_omega_valid) > 0 and gt_c1 != 0:
            comparison["relative_errors"]["c_omega_rel_error"] = float(
                abs(c_omega_valid.mean() - gt_c1) / abs(gt_c1))
        if len(c_theta_valid) > 0 and gt_c2 != 0:
            comparison["relative_errors"]["c_theta_rel_error"] = float(
                abs(c_theta_valid.mean() - gt_c2) / abs(gt_c2))
        if len(intercept_valid) > 0 and gt_dp != 0:
            comparison["relative_errors"]["intercept_abs_rel_error"] = float(
                abs(intercept_valid.abs().mean() - gt_dp) / abs(gt_dp))

        stats["ground_truth_comparison"] = comparison

    stats["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # =================================================================
    # Print Summary
    # =================================================================
    W = 75
    print(f"\n{'=' * W}")
    print(f"AGGREGATED RESULTS - SYNTHETIC NOISELESS 1H - {stats['successful_chunks_gp']} CHUNKS")
    print(f"{'=' * W}")
    print(f"Success rate (GP):  {stats['success_rate_gp']*100:.1f}%  ({stats['successful_chunks_gp']}/{stats['total_chunks']})")
    print(f"Success rate (Sim): {stats['success_rate_sim']*100:.1f}%  ({stats['successful_chunks_sim']}/{stats['total_chunks']})")

    print(f"\nGP State-Estimation RMSE Omega (rad/s):")
    s = stats['orig_rmse_omega']
    print(f"  Mean:   {s['mean']:.6f} +/- {s['std']:.6f}")
    print(f"  Median: {s['median']:.6f}")
    print(f"  IQR:    [{s['q25']:.6f}, {s['q75']:.6f}]")

    print(f"\nForward-Simulated ODE RMSE Omega (rad/s):")
    s = stats['sim_rmse_omega']
    if s.get('count', 0) > 0:
        print(f"  Mean:   {s['mean']:.6f} +/- {s['std']:.6f}")
        print(f"  Median: {s['median']:.6f}")
        print(f"  IQR:    [{s['q25']:.6f}, {s['q75']:.6f}]")
        print(f"  Range:  [{s['min']:.6f}, {s['max']:.6f}]")
    else:
        print(f"  No valid simulation results.")

    print(f"\nGP RMSE Theta (rad):")
    s = stats['orig_rmse_theta']
    print(f"  Mean:   {s['mean']:.6f} +/- {s['std']:.6f}")

    print(f"\nLoss (-ELBO) [finite only: {stats['chunks_with_finite_loss']}/{stats['successful_chunks_gp']}]:")
    s = stats['loss']
    if s.get('count', 0) > 0:
        print(f"  Mean:   {s['mean']:.4f} +/- {s['std']:.4f}")
        print(f"  Median: {s['median']:.4f}")
    else:
        print(f"  No finite loss values.")

    # Coefficient comparison
    print(f"\n{'=' * W}")
    print("RECOVERED PHYSICAL COEFFICIENTS (omega equation)")
    print(f"  domega/dt = c_omega * omega + c_theta * theta + intercept")
    print(f"{'=' * W}")
    print(f"{'Parameter':<25} {'Ground Truth':>15} {'Recovered Mean':>15} {'Recovered Std':>15} {'Rel. Error':>12}")
    print(f"{'-' * W}")

    if ground_truth:
        gt_c1 = ground_truth.get("c_1", float('nan'))
        gt_c2 = ground_truth.get("c_2", float('nan'))
        gt_dp = ground_truth.get("Delta_P", float('nan'))

        c_omega_s = stats["coefficients"]["c_omega"]
        c_theta_s = stats["coefficients"]["c_theta"]
        intercept_s = stats["coefficients"]["intercept"]

        def fmt_rel(recovered_mean, gt_val):
            if gt_val != 0 and not np.isnan(gt_val) and recovered_mean is not None:
                return f"{abs(recovered_mean - gt_val) / abs(gt_val) * 100:.1f}%"
            return "N/A"

        if c_omega_s.get('count', 0) > 0:
            print(f"{'c_omega (= c_1)':<25} {gt_c1:>15.8e} {c_omega_s['mean']:>15.8e} {c_omega_s['std']:>15.8e} {fmt_rel(c_omega_s['mean'], gt_c1):>12}")
        if c_theta_s.get('count', 0) > 0:
            print(f"{'c_theta (= c_2)':<25} {gt_c2:>15.8e} {c_theta_s['mean']:>15.8e} {c_theta_s['std']:>15.8e} {fmt_rel(c_theta_s['mean'], gt_c2):>12}")
        if intercept_s.get('count', 0) > 0:
            abs_mean_int = float(valid["Parsed_intercept"].dropna().abs().mean())
            print(f"{'|intercept| (~ Delta_P)':<25} {gt_dp:>15.8e} {abs_mean_int:>15.8e} {intercept_s['std']:>15.8e} {fmt_rel(abs_mean_int, gt_dp):>12}")
            print(f"{'intercept (signed mean)':<25} {'N/A':>15} {intercept_s['mean']:>15.8e} {intercept_s['std']:>15.8e} {'':>12}")

    # Sample physical equations
    print(f"\n{'=' * W}")
    print("SAMPLE PHYSICAL EQUATIONS (first 10 valid chunks)")
    print(f"{'=' * W}")
    eq_valid = valid[valid["Eq_Omega_Physical"].notna() & (valid["Eq_Omega_Physical"] != "nan")]
    for i, (_, row) in enumerate(eq_valid.head(10).iterrows()):
        chunk_id = int(row["Active_Chunk_Index"])
        orig_id = int(row["Original_Chunk_Index"])
        eq = row["Eq_Omega_Physical"]
        sim_rmse = row["Sim_RMSE_Omega"]
        sim_str = f"{sim_rmse:.6f}" if pd.notna(sim_rmse) else "FAILED"
        print(f"  Chunk {chunk_id:4d} (orig {orig_id:5d}): domega/dt = {eq}  [Sim RMSE: {sim_str}]")

    print(f"\n{'=' * W}")

    # Save outputs
    combined_csv_path = os.path.join(results_dir, "all_chunks_combined.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"\nCombined CSV: {combined_csv_path}")

    stats_path = os.path.join(results_dir, "validation_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Validation summary: {stats_path}")


if __name__ == "__main__":
    main()
