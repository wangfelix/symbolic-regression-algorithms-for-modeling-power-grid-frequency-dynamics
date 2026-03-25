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
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True, help="Folder name inside results_5min_all_chunks, e.g. run_SLURM_12345")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", args.run_name)

    # Find all per-chunk CSV files
    csv_files = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {os.path.basename(f)}")

    # Expected column names
    expected_columns = [
        "Chunk_Index", "Chunk_Start_Time",
        "Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
        "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
        "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
        "Eq_Theta", "Eq_Omega", "Eq_Omega_Physical"
    ]

    # Combine all CSVs
    dfs = []
    for f in csv_files:
        # Peek at first line to check if file has a header row
        with open(f, 'r') as fh:
            first_line = fh.readline().strip()
        has_header = first_line.startswith("Chunk_Index")
        if has_header:
            df = pd.read_csv(f)
        else:
            df = pd.read_csv(f, header=None, names=expected_columns)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows (header={'yes' if has_header else 'no'})")

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case of re-runs)
    combined = combined.drop_duplicates(subset=["Chunk_Index"], keep="last")
    combined = combined.sort_values("Chunk_Index").reset_index(drop=True)

    print(f"\nTotal unique chunks: {len(combined)}")

    # Convert numeric columns
    for col in ["Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
                "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total", "Final_Loss",
                "Diffusion_Theta", "Diffusion_Omega"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Filter valid results
    valid = combined[combined["Orig_RMSE_Omega"].notna()]
    failed = combined[combined["Orig_RMSE_Omega"].isna()]
    n_sim_valid = int(valid["Sim_RMSE_Omega"].notna().sum())

    print(f"Successful (GP): {len(valid)}")
    print(f"Successful (Sim): {n_sim_valid}")
    print(f"Failed: {len(failed)}")

    if len(valid) == 0:
        print("No valid results to aggregate.")
        return

    # Compute statistics
    stats = {
        "total_chunks": int(len(combined)),
        "successful_chunks": int(len(valid)),
        "failed_chunks": int(len(failed)),
        "success_rate_gp": float(len(valid) / len(combined)),
        "successful_chunks_sim": n_sim_valid,
        "success_rate_sim": float(n_sim_valid / len(combined)),
        "orig_rmse_omega": {
            "mean": float(valid["Orig_RMSE_Omega"].mean()),
            "std": float(valid["Orig_RMSE_Omega"].std()),
            "median": float(valid["Orig_RMSE_Omega"].median()),
            "min": float(valid["Orig_RMSE_Omega"].min()),
            "max": float(valid["Orig_RMSE_Omega"].max()),
            "q25": float(valid["Orig_RMSE_Omega"].quantile(0.25)),
            "q75": float(valid["Orig_RMSE_Omega"].quantile(0.75)),
        },
        "sim_rmse_omega": {
            "mean": float(valid["Sim_RMSE_Omega"].mean()),
            "std": float(valid["Sim_RMSE_Omega"].std()),
            "median": float(valid["Sim_RMSE_Omega"].median()),
            "min": float(valid["Sim_RMSE_Omega"].min()),
            "max": float(valid["Sim_RMSE_Omega"].max()),
            "q25": float(valid["Sim_RMSE_Omega"].quantile(0.25)),
            "q75": float(valid["Sim_RMSE_Omega"].quantile(0.75)),
        },
        "orig_rmse_theta": {
            "mean": float(valid["Orig_RMSE_Theta"].mean()),
            "std": float(valid["Orig_RMSE_Theta"].std()),
            "median": float(valid["Orig_RMSE_Theta"].median()),
        },
        "orig_rmse_total": {
            "mean": float(valid["Orig_RMSE_Total"].mean()),
            "std": float(valid["Orig_RMSE_Total"].std()),
            "median": float(valid["Orig_RMSE_Total"].median()),
        },
        "loss": {
            "mean": float(valid["Final_Loss"].mean()),
            "std": float(valid["Final_Loss"].std()),
            "median": float(valid["Final_Loss"].median()),
        },
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Add diffusion stats if columns exist
    if "Diffusion_Omega" in valid.columns:
        diff_omega_valid = valid["Diffusion_Omega"].dropna()
        diff_theta_valid = valid["Diffusion_Theta"].dropna()
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

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"AGGREGATED RESULTS OVER {stats['successful_chunks']} CHUNKS")
    print(f"{'=' * 60}")
    print(f"Success rate (GP):  {stats['success_rate_gp']*100:.1f}%")
    print(f"Success rate (Sim): {stats['success_rate_sim']*100:.1f}%")
    print(f"")
    print(f"GP State-Estimation RMSE Omega (rad/s):")
    print(f"  Mean:   {stats['orig_rmse_omega']['mean']:.6f} ± {stats['orig_rmse_omega']['std']:.6f}")
    print(f"  Median: {stats['orig_rmse_omega']['median']:.6f}")
    print(f"")
    print(f"Forward-Simulated ODE RMSE Omega (rad/s):")
    print(f"  Mean:   {stats['sim_rmse_omega']['mean']:.6f} ± {stats['sim_rmse_omega']['std']:.6f}")
    print(f"  Median: {stats['sim_rmse_omega']['median']:.6f}")
    print(f"  Range:  [{stats['sim_rmse_omega']['min']:.6f}, {stats['sim_rmse_omega']['max']:.6f}]")
    print(f"")
    print(f"RMSE Theta (phase, rad):")
    print(f"  Mean:   {stats['orig_rmse_theta']['mean']:.6f} ± {stats['orig_rmse_theta']['std']:.6f}")
    print(f"")
    print(f"RMSE Total (combined):")
    print(f"  Mean:   {stats['orig_rmse_total']['mean']:.6f} ± {stats['orig_rmse_total']['std']:.6f}")
    print(f"")
    print(f"Loss:")
    print(f"  Mean:   {stats['loss']['mean']:.4f} ± {stats['loss']['std']:.4f}")

    if "diffusion_omega" in stats:
        print(f"")
        print(f"Diffusion (process noise, physical units):")
        d = stats["diffusion_omega"]
        print(f"  Omega: mean={d['mean']:.6e} ± {d['std']:.6e}, median={d['median']:.6e} ({d['count']} valid)")
    if "diffusion_theta" in stats:
        d = stats["diffusion_theta"]
        print(f"  Theta: mean={d['mean']:.6e} ± {d['std']:.6e}, median={d['median']:.6e} ({d['count']} valid)")
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
