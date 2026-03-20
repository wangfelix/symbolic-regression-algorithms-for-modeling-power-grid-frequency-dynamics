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
            # Securely embed the literal file identifier to inherently bypass SLURM index mapping errors
            base = os.path.basename(f)
            task_id_str = base.replace("hp_tuning_combo_", "").replace(".csv", "")
            df["Task_ID"] = int(task_id_str)
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

    # Find best model evaluating strictly empirical Mean_RMSE_Omega instead of the volatile Neural ELBO Loss
    if "Mean_RMSE_Omega" in combined_df.columns and "Mean_Loss" in combined_df.columns:
        combined_df["Mean_RMSE_Omega"] = pd.to_numeric(combined_df["Mean_RMSE_Omega"], errors="coerce")
        combined_df["Mean_Loss"] = pd.to_numeric(combined_df["Mean_Loss"], errors="coerce")
        
        # CRITICAL: We must heavily filter any PyTorch Gradient Explosions (Loss < -50000)
        # Even if the RMSE looks phenomenally small, a blown-up SDE Loss means the mathematical Equations degenerated linearly!
        mask = (combined_df["Mean_Loss"] > -50000) & (combined_df["Mean_Loss"] < 50000)
        
        # Only consider chunks that reliably succeeded (e.g., > 50% success rate to avoid statistical flukes)
        if "Num_Success" in combined_df.columns and "Num_Total" in combined_df.columns:
            combined_df["Num_Success"] = pd.to_numeric(combined_df["Num_Success"], errors="coerce")
            combined_df["Num_Total"] = pd.to_numeric(combined_df["Num_Total"], errors="coerce")
            mask = mask & (combined_df["Num_Success"] > (combined_df["Num_Total"] * 0.5))
            
        valid_df = combined_df[mask]
            
        if len(valid_df) == 0:
            valid_df = combined_df
            
        best_row = valid_df.loc[valid_df["Mean_RMSE_Omega"].idxmin()]
        
        print("\n" + "="*60)
        print("BEST OVERALL HYPERPARAMETER COMBINATION (Minimum Valid RMSE)")
        print("="*60)
        print(best_row)
        print("="*60)
        
        # Copy the best json over explicitly utilizing the mapped File ID physically!
        task_idx = int(best_row["Task_ID"])
        json_path = os.path.join(results_dir, f"combo_{task_idx:03d}.json")
        if os.path.exists(json_path):
            best_json_path = os.path.join(results_dir, "best_overall.json")
            shutil.copy(json_path, best_json_path)
            print(f"\nCopied the best model JSON configuration to:\n{best_json_path}")
    else:
        print("Mean_Loss column not found in merged CSV. Cannot determine best model.")

if __name__ == "__main__":
    main()
