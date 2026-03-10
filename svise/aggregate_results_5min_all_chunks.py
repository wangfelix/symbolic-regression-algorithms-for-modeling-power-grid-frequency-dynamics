"""
Aggregate results from all SLURM array job outputs.

Reads all per-chunk CSV files from results_5min_all_chunks/, combines them,
and computes overall statistics.

Usage:
    python aggregate_results_5min_all_chunks.py
"""
import os
import glob
import pandas as pd
import numpy as np
import json
import datetime


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks")

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
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of re-runs)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"\nTotal unique chunks: {len(combined)}")

    # Convert numeric columns
    for col in ["RMSE_Omega", "RMSE_Theta", "RMSE_Total", "Final_Loss"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results
    valid = combined[combined["RMSE_Omega"].notna()]
    failed = combined[combined["RMSE_Omega"].isna()]

    print(f"Successful: {len(valid)}")
    print(f"Failed: {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # Compute statistics
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
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"AGGREGATED RESULTS OVER {stats['successful_chunks']} CHUNKS")
    print(f"{'=' * 60}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"")
    print(f"RMSE Omega (frequency deviation, rad/s):")
    print(f"  Mean:   {stats['rmse_omega']['mean']:.6f} ± {stats['rmse_omega']['std']:.6f}")
    print(f"  Median: {stats['rmse_omega']['median']:.6f}")
    print(f"  IQR:    [{stats['rmse_omega']['q25']:.6f}, {stats['rmse_omega']['q75']:.6f}]")
    print(f"  Range:  [{stats['rmse_omega']['min']:.6f}, {stats['rmse_omega']['max']:.6f}]")
    print(f"")
    print(f"RMSE Theta (phase, rad):")
    print(f"  Mean:   {stats['rmse_theta']['mean']:.6f} ± {stats['rmse_theta']['std']:.6f}")
    print(f"")
    print(f"RMSE Total (combined):")
    print(f"  Mean:   {stats['rmse_total']['mean']:.6f} ± {stats['rmse_total']['std']:.6f}")
    print(f"")
    print(f"Loss:")
    print(f"  Mean:   {stats['loss']['mean']:.4f} ± {stats['loss']['std']:.4f}")
    print(f"{'=' * 60}")

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
