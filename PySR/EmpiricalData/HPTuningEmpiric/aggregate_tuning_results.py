"""
Aggregate PySR Hyperparameter Tuning Results
=============================================
Hardcoded for:
    /home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/results_hp_tuning/5min

Usage:
    python aggregate_tuning_results.py

What it does:
    1. Reads all hp_tuning_combo_XXX.csv files from the results directory
    2. Combines them into a single results DataFrame
    3. Determines the best combo by lowest Mean_RMSE
    4. Saves full results as all_results.csv and best as best_hyperparams_final.json
    5. Prints a summary table
"""

import os
import json
import datetime

import numpy as np
import pandas as pd


# ── Hardcoded results directory ───────────────────────────────────────────────
RESULTS_DIR = "/home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/results_hp_tuning/5min"
EXPECTED_COMBOS = 60
TOP_N = 10
# ─────────────────────────────────────────────────────────────────────────────


def load_all_csvs(results_dir: str) -> pd.DataFrame:
    """
    Load all hp_tuning_combo_XXX.csv files from the results directory
    and concatenate them into a single DataFrame.
    """
    csv_files = sorted([
        f for f in os.listdir(results_dir)
        if f.startswith("hp_tuning_combo_") and f.endswith(".csv")
    ])

    if not csv_files:
        raise FileNotFoundError(
            f"No hp_tuning_combo_XXX.csv files found in {results_dir}.\n"
            "Make sure all SLURM jobs have finished."
        )

    print(f"Found {len(csv_files)} result files.")

    dfs = []
    missing = []
    for f in csv_files:
        path = os.path.join(results_dir, f)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")
            missing.append(f)

    if missing:
        print(f"\nWARNING: {len(missing)} file(s) could not be read:")
        for f in missing:
            print(f"  {f}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} result rows total.")
    return combined


def check_completeness(df: pd.DataFrame, expected_combos: int = EXPECTED_COMBOS) -> None:
    """Warn if fewer combos than expected are present."""
    n_found = len(df)
    if n_found < expected_combos:
        print(f"\nWARNING: Expected {expected_combos} combos but only found "
              f"{n_found}. Some SLURM jobs may not have finished yet.")
    else:
        print(f"All {n_found} combos present.")


def find_best(df: pd.DataFrame) -> pd.Series:
    """Return the row with the lowest Mean_RMSE (ignoring NaN)."""
    valid = df[df["Mean_RMSE"].notna() & (df["Mean_RMSE"] != "nan")]
    valid = valid.copy()
    valid["Mean_RMSE"] = pd.to_numeric(valid["Mean_RMSE"], errors="coerce")
    valid = valid.dropna(subset=["Mean_RMSE"])

    if valid.empty:
        raise ValueError("No valid (non-NaN) results found across all combo files.")

    best_idx = valid["Mean_RMSE"].idxmin()
    return valid.loc[best_idx]


def print_summary(df: pd.DataFrame, best: pd.Series, top_n: int = TOP_N) -> None:
    """Print a summary table of the top N configurations."""
    df = df.copy()
    df["Mean_RMSE"] = pd.to_numeric(df["Mean_RMSE"], errors="coerce")
    top = df.dropna(subset=["Mean_RMSE"]).nsmallest(top_n, "Mean_RMSE")

    print(f"\n{'=' * 60}")
    print(f"TOP {top_n} CONFIGURATIONS (by Mean RMSE)")
    print(f"{'=' * 60}")
    print(top.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("BEST CONFIGURATION")
    print(f"{'=' * 60}")
    for col, val in best.items():
        print(f"  {col:<26} {val}")
    print(f"{'=' * 60}")


def save_results(df: pd.DataFrame,
                 best: pd.Series,
                 results_dir: str) -> None:
    """Save full results CSV and best hyperparams JSON."""

    # Full sorted results
    df_sorted = df.copy()
    df_sorted["Mean_RMSE"] = pd.to_numeric(df_sorted["Mean_RMSE"], errors="coerce")
    df_sorted = df_sorted.sort_values("Mean_RMSE").reset_index(drop=True)
    all_csv_path = os.path.join(results_dir, "all_results.csv")
    df_sorted.to_csv(all_csv_path, index=False)
    print(f"\nFull results saved to: {all_csv_path}")

    # Best hyperparams as JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_json_path = os.path.join(results_dir, "best_hyperparams_final.json")

    # Identify hyperparameter columns (everything except metadata columns)
    meta_cols = {"Combo_Index", "Mean_RMSE", "Num_Success", "Num_Total"}
    hp_cols   = [c for c in best.index if c not in meta_cols]

    best_dict = {
        "best_hyperparams": {k: best[k] for k in hp_cols},
        "mean_rmse"        : float(best["Mean_RMSE"]),
        "combo_index"      : int(best["Combo_Index"]) if "Combo_Index" in best.index else None,
        "num_success"      : int(best["Num_Success"]) if "Num_Success" in best.index else None,
        "num_total"        : int(best["Num_Total"])   if "Num_Total"   in best.index else None,
        "n_combos_evaluated": len(df),
        "timestamp"        : timestamp,
    }

    with open(best_json_path, "w") as f:
        json.dump(best_dict, f, indent=4)
    print(f"Best hyperparams saved to: {best_json_path}")


def main():
    results_dir = RESULTS_DIR

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Results directory: {results_dir}")

    # 1. Load all CSVs
    df = load_all_csvs(results_dir)

    # 2. Check completeness
    check_completeness(df, expected_combos=EXPECTED_COMBOS)

    # 3. Find best
    best = find_best(df)

    # 4. Print summary
    print_summary(df, best, top_n=TOP_N)

    # 5. Save
    save_results(df, best, results_dir)


if __name__ == "__main__":
    main()
