"""
Aggregate results from SINDy SLURM array job outputs.

Reads all per-chunk CSV files from results_sindy_5min_all_chunks/<run-name>/,
combines them, and computes overall statistics.

Usage:
    python aggregate_results_sindy.py --run-name run_SLURM_12345_sindy
"""
import os
import glob
import pandas as pd
import numpy as np
import json
import datetime
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True,
                        help="Folder name inside results_sindy_5min_all_chunks")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__),
                               "results_sindy_5min_all_chunks", args.run_name)

    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    # Combine all CSVs
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
                "Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
            ])
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"Total unique chunks: {len(combined)}")

    # Convert numeric columns
    numeric_cols = ["Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
                    "Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                    "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results (simulation succeeded)
    valid = combined[combined["Sim_RMSE_Omega"].notna()
                     & np.isfinite(combined["Sim_RMSE_Omega"])]
    failed = combined[~combined.index.isin(valid.index)]

    print(f"Successful (Sim): {len(valid)}")
    print(f"Failed: {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # Compute statistics
    stats = {
        "total_chunks": int(len(combined)),
        "successful_chunks_sim": int(len(valid)),
        "failed_chunks": int(len(failed)),
        "success_rate_sim": float(len(valid) / len(combined)),
        "sim_rmse_omega": {
            "mean": float(valid["Sim_RMSE_Omega"].mean()),
            "std": float(valid["Sim_RMSE_Omega"].std()),
            "median": float(valid["Sim_RMSE_Omega"].median()),
            "min": float(valid["Sim_RMSE_Omega"].min()),
            "max": float(valid["Sim_RMSE_Omega"].max()),
            "q25": float(valid["Sim_RMSE_Omega"].quantile(0.25)),
            "q75": float(valid["Sim_RMSE_Omega"].quantile(0.75)),
        },
        "sim_rmse_theta": {
            "mean": float(valid["Sim_RMSE_Theta"].mean()),
            "std": float(valid["Sim_RMSE_Theta"].std()),
            "median": float(valid["Sim_RMSE_Theta"].median()),
        },
        "sim_rmse_total": {
            "mean": float(valid["Sim_RMSE_Total"].mean()),
            "std": float(valid["Sim_RMSE_Total"].std()),
            "median": float(valid["Sim_RMSE_Total"].median()),
        },
        "config": {
            "sigma": 15,
            "degree": 2,
            "stlsq_threshold": 1e-10,
            "features": ["theta", "omega"],
        },
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SINDy AGGREGATED RESULTS — {stats['successful_chunks_sim']} CHUNKS")
    print(f"{'=' * 60}")
    print(f"Success rate (Sim): {stats['success_rate_sim']*100:.1f}%")
    print(f"")
    print(f"Forward-Simulated RMSE Omega (rad/s):")
    s = stats['sim_rmse_omega']
    print(f"  Mean:   {s['mean']:.6f} ± {s['std']:.6f}")
    print(f"  Median: {s['median']:.6f}")
    print(f"  Range:  [{s['min']:.6f}, {s['max']:.6f}]")
    print(f"  IQR:    [{s['q25']:.6f}, {s['q75']:.6f}]")
    print(f"")
    print(f"RMSE Theta (phase, rad):")
    print(f"  Mean:   {stats['sim_rmse_theta']['mean']:.6f} ± {stats['sim_rmse_theta']['std']:.6f}")
    print(f"{'=' * 60}")

    # Save
    combined_csv_path = os.path.join(results_dir, "all_chunks_combined.csv")
    combined.to_csv(combined_csv_path, index=False)
    print(f"\nCombined CSV saved to: {combined_csv_path}")

    stats_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Aggregated summary saved to: {stats_path}")


if __name__ == "__main__":
    main()
