"""
Find chunks with the best (lowest) forward-simulated ODE RMSE.

Reads the combined CSV from a SVISE run and ranks chunks by Sim_RMSE_Omega.

Usage:
    python find_best_ode_chunks.py --run-dir results_5min_all_chunks/run_SLURM_3718912_combo5
    python find_best_ode_chunks.py --run-dir results_5min_all_chunks/run_SLURM_3718868_combo38 --top 20
"""
import os
import argparse
import numpy as np
import pandas as pd
import glob


def main():
    parser = argparse.ArgumentParser(description="Find chunks with best ODE simulation RMSE")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory containing chunk CSVs or all_chunks_combined.csv")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top chunks to display")
    parser.add_argument("--sort-by", type=str, default="Sim_RMSE_Omega",
                        choices=["Sim_RMSE_Omega", "Orig_RMSE_Omega", "Sim_RMSE_Total"],
                        help="Column to sort by")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(os.path.dirname(__file__), run_dir)

    # Try combined CSV first, then individual chunk CSVs
    combined_path = os.path.join(run_dir, "all_chunks_combined.csv")
    if os.path.exists(combined_path):
        print(f"Reading: {combined_path}")
        df = pd.read_csv(combined_path)
    else:
        csv_files = sorted(glob.glob(os.path.join(run_dir, "chunks_*.csv")))
        if not csv_files:
            print(f"No CSV files found in {run_dir}")
            return
        print(f"Reading {len(csv_files)} CSV files from {run_dir}")
        dfs = []
        for f in csv_files:
            dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["Chunk_Index"], keep="last")

    # Convert numeric columns
    for col in ["Orig_RMSE_Omega", "Sim_RMSE_Omega", "Sim_RMSE_Total", "Final_Loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_col = args.sort_by
    if sort_col not in df.columns:
        print(f"Column '{sort_col}' not found. Available: {list(df.columns)}")
        return

    # Filter to valid sim results
    valid = df[df[sort_col].notna()].copy()
    print(f"Total rows: {len(df)}, Valid ({sort_col}): {len(valid)}")

    if len(valid) == 0:
        print("No valid results.")
        return

    # Sort
    valid = valid.sort_values(sort_col).reset_index(drop=True)

    # Print top N
    W = 120
    print(f"\n{'=' * W}")
    print(f"  TOP {args.top} CHUNKS BY {sort_col} (lower = better)")
    print(f"{'=' * W}")

    has_start_time = "Chunk_Start_Time" in valid.columns
    has_phys_eq = "Eq_Omega_Physical" in valid.columns

    header = f"{'Rank':>4}  {'Chunk':>6}"
    if has_start_time:
        header += f"  {'Start Time':<22}"
    header += f"  {'GP RMSE ω':>10}  {'Sim RMSE ω':>11}  {'Loss':>12}"
    if has_phys_eq:
        header += f"  {'Physical Equation'}"
    print(header)
    print(f"{'-' * W}")

    for rank, (_, row) in enumerate(valid.head(args.top).iterrows(), 1):
        chunk_col = "Chunk_Index" if "Chunk_Index" in row.index else "Active_Chunk_Index"
        line = f"{rank:>4}  {int(row[chunk_col]):>6}"
        if has_start_time:
            line += f"  {str(row['Chunk_Start_Time']):<22}"
        orig = row.get("Orig_RMSE_Omega", np.nan)
        sim = row.get("Sim_RMSE_Omega", np.nan)
        loss = row.get("Final_Loss", np.nan)
        line += f"  {orig:>10.6f}  {sim:>11.6f}  {loss:>12.4f}"
        if has_phys_eq:
            eq = row.get("Eq_Omega_Physical", "N/A")
            line += f"  {eq}"
        print(line)

    # Statistics
    print(f"\n{'=' * W}")
    print(f"  STATISTICS FOR {sort_col}")
    print(f"{'=' * W}")
    s = valid[sort_col]
    print(f"  Mean:   {s.mean():.6f} ± {s.std():.6f}")
    print(f"  Median: {s.median():.6f}")
    print(f"  Min:    {s.min():.6f}")
    print(f"  Max:    {s.max():.6f}")
    print(f"  IQR:    [{s.quantile(0.25):.6f}, {s.quantile(0.75):.6f}]")
    print(f"  Count:  {len(s)}")
    print(f"{'=' * W}")

    # Print best chunk info
    best = valid.iloc[0]
    chunk_col = "Chunk_Index" if "Chunk_Index" in best.index else "Active_Chunk_Index"
    print(f"\nBest chunk: {int(best[chunk_col])} with {sort_col} = {best[sort_col]:.6f}")


if __name__ == "__main__":
    main()
