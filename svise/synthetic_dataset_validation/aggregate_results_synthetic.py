"""
Aggregate results from SVISE evaluation on synthetic noiseless dataset.

Reads all per-chunk CSV files from results_synthetic_noiseless/, combines them,
computes overall statistics, and compares recovered coefficients to ground truth.

Usage:
    python aggregate_results_synthetic.py
"""
import os
import glob
import pandas as pd
import numpy as np
import json
import datetime


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results_synthetic_noiseless")

    # Find all per-chunk CSV files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    # Combine all CSVs
    dfs = []
    for f in csv_files:
        with open(f, 'r') as fh:
            first_line = fh.readline().strip()
        has_header = first_line.startswith("Chunk_Index")
        if has_header:
            df = pd.read_csv(f)
        else:
            # Fallback: read without header (shouldn't happen normally)
            df = pd.read_csv(f, header=None)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of re-runs)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"\nTotal unique chunks: {len(combined)}")

    # Convert numeric columns
    numeric_cols = ["RMSE_Omega", "RMSE_Theta", "RMSE_Total", "Final_Loss"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Also convert coefficient columns
    coeff_cols = [c for c in combined.columns if c.startswith("Coeff_")]
    for col in coeff_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Convert diffusion columns
    diff_cols = [c for c in combined.columns if c.startswith("Diffusion_")]
    for col in diff_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results
    valid = combined[combined["RMSE_Omega"].notna()]
    failed = combined[combined["RMSE_Omega"].isna()]

    print(f"Successful: {len(valid)}")
    print(f"Failed: {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # Load ground truth parameters
    gt_path = os.path.join(script_dir, "ground_truth_params.json")
    ground_truth = None
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
        print(f"\nGround truth loaded from: {gt_path}")
    else:
        print(f"\nWarning: Ground truth not found at {gt_path}")

    # =================================================================
    # RMSE Statistics
    # =================================================================
    stats = {
        "total_chunks": int(len(combined)),
        "successful_chunks": int(len(valid)),
        "failed_chunks": int(len(failed)),
        "success_rate": float(len(valid) / len(combined)),
        "rmse_omega": {
            "mean": float(valid["RMSE_Omega"].mean()),
            "std": float(valid["RMSE_Omega"].std()),
            "median": float(valid["RMSE_Omega"].median()),
            "min": float(valid["RMSE_Omega"].min()),
            "max": float(valid["RMSE_Omega"].max()),
            "q25": float(valid["RMSE_Omega"].quantile(0.25)),
            "q75": float(valid["RMSE_Omega"].quantile(0.75)),
        },
        "rmse_theta": {
            "mean": float(valid["RMSE_Theta"].mean()),
            "std": float(valid["RMSE_Theta"].std()),
            "median": float(valid["RMSE_Theta"].median()),
        },
        "rmse_total": {
            "mean": float(valid["RMSE_Total"].mean()),
            "std": float(valid["RMSE_Total"].std()),
            "median": float(valid["RMSE_Total"].median()),
        },
        "loss": {
            "mean": float(valid["Final_Loss"].mean()),
            "std": float(valid["Final_Loss"].std()),
            "median": float(valid["Final_Loss"].median()),
        },
    }

    # =================================================================
    # Polynomial Coefficient Statistics
    # =================================================================
    coeff_stats = {}
    for col in coeff_cols:
        col_valid = valid[col].dropna()
        if len(col_valid) > 0:
            coeff_stats[col] = {
                "mean": float(col_valid.mean()),
                "std": float(col_valid.std()),
                "median": float(col_valid.median()),
                "min": float(col_valid.min()),
                "max": float(col_valid.max()),
            }

    stats["coefficients"] = coeff_stats

    # =================================================================
    # Diffusion Statistics
    # =================================================================
    diff_stats = {}
    for col in diff_cols:
        col_valid = valid[col].dropna()
        if len(col_valid) > 0:
            diff_stats[col] = {
                "mean": float(col_valid.mean()),
                "std": float(col_valid.std()),
                "median": float(col_valid.median()),
            }

    stats["diffusion"] = diff_stats

    # =================================================================
    # Ground Truth Comparison
    # =================================================================
    if ground_truth:
        comparison = {
            "ground_truth": {
                "c_1": ground_truth.get("c_1"),
                "c_2": ground_truth.get("c_2"),
                "c_2_decay": ground_truth.get("c_2_decay"),
                "Delta_P": ground_truth.get("Delta_P"),
                "epsilon": ground_truth.get("epsilon"),
            },
            "recovered": {},
        }

        # Map SVISE coefficients to physical parameters
        # For degree=1 integrator, omega equation: domega/dt = Intercept + c_theta*theta + c_omega*omega
        # c_omega should match c_1
        # c_theta should match c_2 = c_2_decay * c_1
        # Intercept captures Delta_P * P(t) * sign(t) averaged over chunk
        if "Coeff_Omega" in coeff_stats:
            comparison["recovered"]["c_omega (should match c_1)"] = coeff_stats["Coeff_Omega"]
        if "Coeff_Theta" in coeff_stats:
            comparison["recovered"]["c_theta (should match c_2=c_2_decay*c_1)"] = coeff_stats["Coeff_Theta"]
        if "Coeff_Intercept" in coeff_stats:
            comparison["recovered"]["intercept (captures Delta_P effect)"] = coeff_stats["Coeff_Intercept"]

        # Higher-order coefficients (should be ~0 for noiseless linear data)
        for col in coeff_cols:
            if col not in ["Coeff_Intercept", "Coeff_Theta", "Coeff_Omega"]:
                if col in coeff_stats:
                    comparison["recovered"][f"{col} (should be ~0)"] = coeff_stats[col]

        stats["ground_truth_comparison"] = comparison

    stats["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # =================================================================
    # Print Summary
    # =================================================================
    print(f"\n{'=' * 70}")
    print(f"AGGREGATED RESULTS — SYNTHETIC NOISELESS — {stats['successful_chunks']} CHUNKS")
    print(f"{'=' * 70}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")

    print(f"\nRMSE Omega (rad/s):")
    print(f"  Mean:   {stats['rmse_omega']['mean']:.6f} +/- {stats['rmse_omega']['std']:.6f}")
    print(f"  Median: {stats['rmse_omega']['median']:.6f}")
    print(f"  IQR:    [{stats['rmse_omega']['q25']:.6f}, {stats['rmse_omega']['q75']:.6f}]")

    print(f"\nRMSE Theta (rad):")
    print(f"  Mean:   {stats['rmse_theta']['mean']:.6f} +/- {stats['rmse_theta']['std']:.6f}")

    print(f"\nLoss (-ELBO):")
    print(f"  Mean:   {stats['loss']['mean']:.4f} +/- {stats['loss']['std']:.4f}")

    if coeff_stats:
        print(f"\n{'=' * 70}")
        print("RECOVERED POLYNOMIAL COEFFICIENTS (omega equation)")
        print(f"{'=' * 70}")
        print(f"{'Coefficient':<25} {'Mean':>15} {'Std':>15} {'Median':>15}")
        print(f"{'-' * 70}")
        for col, s in coeff_stats.items():
            label = col.replace("Coeff_", "")
            print(f"{label:<25} {s['mean']:>15.8e} {s['std']:>15.8e} {s['median']:>15.8e}")

    if ground_truth:
        print(f"\n{'=' * 70}")
        print("COMPARISON: RECOVERED vs GROUND TRUTH")
        print(f"{'=' * 70}")
        print(f"{'Parameter':<30} {'Ground Truth':>15} {'Recovered Mean':>15} {'Recovered Std':>15}")
        print(f"{'-' * 75}")

        gt_c1 = ground_truth.get("c_1", float('nan'))
        gt_c2 = ground_truth.get("c_2", float('nan'))
        gt_dp = ground_truth.get("Delta_P", float('nan'))

        rec_omega = coeff_stats.get("Coeff_Omega", {})
        rec_theta = coeff_stats.get("Coeff_Theta", {})
        rec_intercept = coeff_stats.get("Coeff_Intercept", {})

        print(f"{'c_1 (omega coeff)':<30} {gt_c1:>15.8e} "
              f"{rec_omega.get('mean', float('nan')):>15.8e} "
              f"{rec_omega.get('std', float('nan')):>15.8e}")
        print(f"{'c_2 (theta coeff)':<30} {gt_c2:>15.8e} "
              f"{rec_theta.get('mean', float('nan')):>15.8e} "
              f"{rec_theta.get('std', float('nan')):>15.8e}")
        print(f"{'Delta_P (intercept)':<30} {gt_dp:>15.8e} "
              f"{rec_intercept.get('mean', float('nan')):>15.8e} "
              f"{rec_intercept.get('std', float('nan')):>15.8e}")

        # Higher-order terms
        for col in coeff_cols:
            if col not in ["Coeff_Intercept", "Coeff_Theta", "Coeff_Omega"] and col in coeff_stats:
                label = col.replace("Coeff_", "")
                s = coeff_stats[col]
                print(f"{label + ' (should be ~0)':<30} {'0':>15} "
                      f"{s['mean']:>15.8e} {s['std']:>15.8e}")

    if diff_stats:
        print(f"\n{'=' * 70}")
        print("DIFFUSION / NOISE (should be ~0 for noiseless data)")
        print(f"{'=' * 70}")
        for col, s in diff_stats.items():
            print(f"  {col}: mean={s['mean']:.8e}, std={s['std']:.8e}")

    print(f"\n{'=' * 70}")

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
