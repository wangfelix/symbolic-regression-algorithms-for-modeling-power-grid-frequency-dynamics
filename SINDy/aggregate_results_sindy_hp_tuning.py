"""
Aggregate results from SINDy hyperparameter tuning SLURM array job.

Reads all per-combo CSV files from results_sindy_hp_tuning/<run-name>/,
merges them into a single master CSV, identifies the best hyperparameter
combination by minimum Mean_RMSE_Omega, and saves an aggregated summary.

Usage:
    python aggregate_results_sindy_hp_tuning.py --run-name run_SLURM_3754208
"""
import os
import glob
import pandas as pd
import numpy as np
import argparse
import json
import shutil
import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate parallel SLURM array SINDy HP tuning runs."
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Folder name inside results_sindy_hp_tuning, e.g. run_SLURM_3754208"
    )
    args = parser.parse_args()

    results_dir = os.path.join(
        os.path.dirname(__file__), "results_sindy_hp_tuning", args.run_name
    )
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    # ---- Collect per-combo CSVs ----
    csv_files = sorted(glob.glob(os.path.join(results_dir, "hp_tuning_combo_*.csv")))
    if not csv_files:
        print(f"No hp_tuning_combo_*.csv files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} combo CSV files.")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Embed the file-based task ID to avoid SLURM index mapping issues
            base = os.path.basename(f)
            task_id_str = base.replace("hp_tuning_combo_", "").replace(".csv", "")
            df["Task_ID"] = int(task_id_str)
            dfs.append(df)
        except Exception as e:
            print(f"  Could not read {f}: {e}")

    if not dfs:
        print("No valid CSV files could be read.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by combo index
    if "Combo_Index" in combined_df.columns:
        combined_df = combined_df.sort_values(by="Combo_Index").reset_index(drop=True)

    # ---- Save master CSV ----
    master_csv_path = os.path.join(results_dir, f"sindy_hp_tuning_{args.run_name}_all.csv")
    combined_df.to_csv(master_csv_path, index=False)
    print(f"Generated master CSV at: {master_csv_path}")
    print(f"Total combos in master CSV: {len(combined_df)}")

    # ---- Convert numeric columns ----
    numeric_cols = [
        "Mean_RMSE_Omega", "Median_RMSE_Omega", "Std_RMSE_Omega",
        "Mean_RMSE_Theta", "Mean_RMSE_Total",
        "Num_Success", "Num_Total",
    ]
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    # ---- Find best combo ----
    # Filter: must have >50% success rate to avoid statistical flukes
    mask = pd.Series(True, index=combined_df.index)

    if "Num_Success" in combined_df.columns and "Num_Total" in combined_df.columns:
        mask = mask & (combined_df["Num_Success"] > (combined_df["Num_Total"] * 0.5))

    # Filter NaN RMSE values
    if "Mean_RMSE_Omega" in combined_df.columns:
        mask = mask & combined_df["Mean_RMSE_Omega"].notna()
        mask = mask & np.isfinite(combined_df["Mean_RMSE_Omega"])

    valid_df = combined_df[mask]

    if len(valid_df) == 0:
        print("Warning: No combos passed the success-rate filter. Using all combos.")
        valid_df = combined_df[
            combined_df["Mean_RMSE_Omega"].notna()
            & np.isfinite(combined_df["Mean_RMSE_Omega"])
        ]

    if len(valid_df) == 0:
        print("No valid results to aggregate.")
        return

    best_row = valid_df.loc[valid_df["Mean_RMSE_Omega"].idxmin()]

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print(f"SINDy HP TUNING — AGGREGATED RESULTS ({len(combined_df)} COMBOS)")
    print(f"{'=' * 60}")
    print(f"Combos with >50% success rate: {len(valid_df)}/{len(combined_df)}")

    # Overall RMSE statistics across all valid combos
    print(f"\nMean_RMSE_Omega across all valid combos:")
    print(f"  Best:   {valid_df['Mean_RMSE_Omega'].min():.6f}")
    print(f"  Worst:  {valid_df['Mean_RMSE_Omega'].max():.6f}")
    print(f"  Mean:   {valid_df['Mean_RMSE_Omega'].mean():.6f}")
    print(f"  Median: {valid_df['Mean_RMSE_Omega'].median():.6f}")

    print(f"\n{'=' * 60}")
    print(f"BEST HYPERPARAMETER COMBINATION (Minimum Mean_RMSE_Omega)")
    print(f"{'=' * 60}")
    print(f"  Combo Index:      {int(best_row.get('Combo_Index', -1))}")
    if "Sigma" in best_row:
        print(f"  Sigma:            {best_row['Sigma']}")
    if "Degree" in best_row:
        print(f"  Degree:           {int(best_row['Degree'])}")
    if "Threshold" in best_row:
        print(f"  Threshold:        {best_row['Threshold']}")
    print(f"  Mean RMSE Omega:  {best_row['Mean_RMSE_Omega']:.6f}")
    if "Median_RMSE_Omega" in best_row:
        print(f"  Median RMSE Omega:{best_row['Median_RMSE_Omega']:.6f}")
    if "Std_RMSE_Omega" in best_row:
        print(f"  Std RMSE Omega:   {best_row['Std_RMSE_Omega']:.6f}")
    if "Num_Success" in best_row and "Num_Total" in best_row:
        pct = best_row['Num_Success'] / best_row['Num_Total'] * 100
        print(f"  Success Rate:     {int(best_row['Num_Success'])}/{int(best_row['Num_Total'])} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    # ---- Top-10 table ----
    top10 = valid_df.nsmallest(10, "Mean_RMSE_Omega")
    print(f"\nTop-10 Combos by Mean_RMSE_Omega:")
    print(f"{'Rank':<6}{'Combo':<8}{'Sigma':<8}{'Degree':<8}{'Threshold':<14}{'Mean_RMSE':<14}{'Median_RMSE':<14}{'Success':<10}")
    print("-" * 82)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        sigma = row.get("Sigma", "?")
        degree = int(row.get("Degree", 0))
        threshold = row.get("Threshold", "?")
        mean_rmse = row["Mean_RMSE_Omega"]
        median_rmse = row.get("Median_RMSE_Omega", float('nan'))
        n_success = int(row.get("Num_Success", 0))
        n_total = int(row.get("Num_Total", 0))
        combo_idx = int(row.get("Combo_Index", -1))

        # Format threshold nicely
        if isinstance(threshold, float):
            threshold_str = f"{threshold:.1e}"
        else:
            threshold_str = str(threshold)

        print(f"{rank:<6}{combo_idx:<8}{sigma:<8}{degree:<8}{threshold_str:<14}{mean_rmse:<14.6f}{median_rmse:<14.6f}{n_success}/{n_total}")

    # ---- Copy best JSON ----
    task_id = int(best_row.get("Task_ID", best_row.get("Combo_Index", -1)))
    json_src = os.path.join(results_dir, f"combo_{task_id:03d}.json")
    if os.path.exists(json_src):
        best_json_dst = os.path.join(results_dir, "best_overall.json")
        shutil.copy(json_src, best_json_dst)
        print(f"\nCopied best combo JSON to: {best_json_dst}")

    # ---- Save aggregated summary ----
    summary = {
        "run_name": args.run_name,
        "total_combos": int(len(combined_df)),
        "valid_combos_above_50pct_success": int(len(valid_df)),
        "best_combo": {
            "combo_index": int(best_row.get("Combo_Index", -1)),
            "sigma": int(best_row.get("Sigma", 0)) if pd.notna(best_row.get("Sigma")) else None,
            "degree": int(best_row.get("Degree", 0)) if pd.notna(best_row.get("Degree")) else None,
            "threshold": float(best_row.get("Threshold", 0)) if pd.notna(best_row.get("Threshold")) else None,
            "mean_rmse_omega": float(best_row["Mean_RMSE_Omega"]),
            "median_rmse_omega": float(best_row.get("Median_RMSE_Omega", float('nan'))),
            "std_rmse_omega": float(best_row.get("Std_RMSE_Omega", float('nan'))),
            "num_success": int(best_row.get("Num_Success", 0)),
            "num_total": int(best_row.get("Num_Total", 0)),
        },
        "rmse_omega_stats_across_combos": {
            "min": float(valid_df["Mean_RMSE_Omega"].min()),
            "max": float(valid_df["Mean_RMSE_Omega"].max()),
            "mean": float(valid_df["Mean_RMSE_Omega"].mean()),
            "median": float(valid_df["Mean_RMSE_Omega"].median()),
            "std": float(valid_df["Mean_RMSE_Omega"].std()),
        },
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    summary_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Aggregated summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
