import os
import glob
import pandas as pd
import argparse
import json
import shutil

def main():
    parser = argparse.ArgumentParser(description="Aggregates parallel SLURM array hyperparameter tuning runs.")
    parser.add_argument("--run-name", type=str, required=True, help="Folder name inside results_hp_tuning, e.g. run_SLURM_12345")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results_hp_tuning", args.run_name)
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    csv_files = glob.glob(os.path.join(results_dir, "hp_tuning_combo_*.csv"))
    if not csv_files:
        print(f"No hp_tuning_combo_*.csv files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} combo execution results.")

    # Read and merge CSVs (some files may have headers, but pandas handles it cleanly on concat)
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not dfs:
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by combo index
    if "Combo_Index" in combined_df.columns:
        combined_df = combined_df.sort_values(by="Combo_Index").reset_index(drop=True)

    # Save master CSV
    master_csv_path = os.path.join(results_dir, f"hp_tuning_{args.run_name}_all.csv")
    combined_df.to_csv(master_csv_path, index=False)
    print(f"Generated master CSV at: {master_csv_path}")

    # Find best model evaluating numerical Mean_Loss
    if "Mean_Loss" in combined_df.columns:
        combined_df["Mean_Loss"] = pd.to_numeric(combined_df["Mean_Loss"], errors="coerce")
        best_row = combined_df.loc[combined_df["Mean_Loss"].idxmin()]
        
        print("\n" + "="*60)
        print("BEST OVERALL HYPERPARAMETER COMBINATION (-ELBO)")
        print("="*60)
        print(best_row)
        print("="*60)
        
        # Copy the best json over
        combo_idx = int(best_row["Combo_Index"])
        json_path = os.path.join(results_dir, f"combo_{combo_idx:03d}.json")
        if os.path.exists(json_path):
            best_json_path = os.path.join(results_dir, "best_overall.json")
            shutil.copy(json_path, best_json_path)
            print(f"\nCopied the best model JSON configuration to:\n{best_json_path}")
    else:
        print("Mean_Loss column not found in merged CSV. Cannot determine best model.")

if __name__ == "__main__":
    main()
