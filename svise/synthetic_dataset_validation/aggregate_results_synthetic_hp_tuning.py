import os
import glob
import pandas as pd
import argparse
import json
import shutil


def print_combo(label, row, columns):
    """Pretty-print a single combo result."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    for col in columns:
        if col in row.index:
            val = row[col]
            if isinstance(val, float):
                if abs(val) > 1e4 or (abs(val) < 1e-3 and val != 0):
                    print(f"  {col:30s}: {val:.6e}")
                else:
                    print(f"  {col:30s}: {val:.6f}")
            else:
                print(f"  {col:30s}: {val}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate synthetic HP tuning results and find best combos."
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Folder name inside results dir, e.g. run_SLURM_3718590"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results_hp_tuning",
        help="Results parent folder name (default: results_hp_tuning). E.g. results_hp_tuning_1h"
    )
    parser.add_argument(
        "--loss-cap", type=float, default=-50000,
        help="Loss cap threshold. Combos with Mean_Loss < this are considered diverged (default: -50000)"
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), args.results_dir, args.run_name)
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        return

    csv_files = sorted(glob.glob(os.path.join(results_dir, "hp_tuning_combo_*.csv")))
    if not csv_files:
        print(f"No hp_tuning_combo_*.csv files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} combo result files.")

    # Read and merge CSVs
    dfs = []
    empty_count = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                empty_count += 1
                continue
            base = os.path.basename(f)
            task_id_str = base.replace("hp_tuning_combo_", "").replace(".csv", "")
            df["Task_ID"] = int(task_id_str)
            dfs.append(df)
        except Exception as e:
            print(f"  Could not read {f}: {e}")

    if not dfs:
        print("No valid results found.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Coerce numeric columns
    for col in ["Mean_RMSE_Omega", "Mean_Sim_RMSE_Omega", "Mean_Loss",
                "Num_Success", "Num_Sim_Success", "Num_Total"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.sort_values(by="Combo_Index").reset_index(drop=True)

    # Save master CSV
    master_csv = os.path.join(results_dir, f"hp_tuning_{args.run_name}_all.csv")
    combined.to_csv(master_csv, index=False)

    # --- Summary statistics ---
    n_total = len(csv_files)
    n_with_results = len(combined)
    n_finite_loss = combined["Mean_Loss"].apply(lambda x: pd.notna(x) and abs(x) < float("inf")).sum()
    n_capped = combined["Mean_Loss"].apply(lambda x: pd.notna(x) and x > args.loss_cap and abs(x) < float("inf")).sum()

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total combo files:          {n_total}")
    print(f"  Empty (no results):         {empty_count}")
    print(f"  With results:               {n_with_results}")
    print(f"  Finite loss:                {n_finite_loss}")
    print(f"  Within loss cap (>{args.loss_cap:.0f}):  {n_capped}")
    if "Num_Sim_Success" in combined.columns:
        n_with_sim = (combined["Num_Sim_Success"] > 0).sum()
        print(f"  With sim RMSE data:         {n_with_sim}")
    print(f"{'='*70}")

    # Display columns for combo printout
    display_cols = [
        "Combo_Index", "Degree", "Tau", "Lr", "N_Tau",
        "Measurement_Noise", "N_Reparam_Samples",
        "Mean_Loss", "Mean_RMSE_Omega", "Mean_Sim_RMSE_Omega",
        "Num_Success", "Num_Sim_Success", "Num_Total"
    ]

    # --- Criterion 1: Lowest loss (uncapped) ---
    # Loss = negative ELBO. Lower (more negative) = better fit.
    finite_mask = combined["Mean_Loss"].apply(lambda x: pd.notna(x) and abs(x) < float("inf"))
    if finite_mask.any():
        df1 = combined[finite_mask]
        best1 = df1.loc[df1["Mean_Loss"].idxmin()]  # Lower loss = better
        print_combo("1) BEST LOSS (uncapped, lowest loss)", best1, display_cols)
    else:
        print("\n  No combos with finite loss found!")

    # --- Criterion 2: Lowest loss with cap ---
    cap_mask = combined["Mean_Loss"].apply(
        lambda x: pd.notna(x) and x > args.loss_cap and abs(x) < float("inf")
    )
    if cap_mask.any():
        df2 = combined[cap_mask]
        best2 = df2.loc[df2["Mean_Loss"].idxmin()]  # Lower loss = better
        print_combo(f"2) BEST LOSS (capped > {args.loss_cap:.0f})", best2, display_cols)
    else:
        print(f"\n  No combos with loss > {args.loss_cap:.0f} found!")

    # --- Criterion 3: Lowest sim RMSE (uncapped) ---
    if "Mean_Sim_RMSE_Omega" in combined.columns:
        sim_mask = combined["Mean_Sim_RMSE_Omega"].apply(
            lambda x: pd.notna(x) and x > 0 and abs(x) < float("inf")
        )
        if sim_mask.any():
            df3 = combined[sim_mask]
            best3 = df3.loc[df3["Mean_Sim_RMSE_Omega"].idxmin()]
            print_combo("3) BEST SIM RMSE (uncapped)", best3, display_cols)
        else:
            print("\n  No combos with valid Sim RMSE found!")
    else:
        print("\n  Mean_Sim_RMSE_Omega column not found.")

    # --- Criterion 4: Lowest sim RMSE with loss cap ---
    if "Mean_Sim_RMSE_Omega" in combined.columns:
        both_mask = cap_mask & sim_mask if sim_mask.any() else cap_mask
        if both_mask.any():
            df4 = combined[both_mask]
            best4 = df4.loc[df4["Mean_Sim_RMSE_Omega"].idxmin()]
            print_combo(f"4) BEST SIM RMSE (loss capped > {args.loss_cap:.0f})", best4, display_cols)
        else:
            print(f"\n  No combos satisfy both sim RMSE valid + loss > {args.loss_cap:.0f}!")

    # --- Criterion 5: Lowest sim RMSE, then lowest loss as tiebreaker ---
    if "Mean_Sim_RMSE_Omega" in combined.columns and sim_mask.any():
        df5 = combined[sim_mask].copy()
        df5 = df5.sort_values(by=["Mean_Sim_RMSE_Omega", "Mean_Loss"], ascending=[True, True])
        best5 = df5.iloc[0]
        print_combo("5) BEST SIM RMSE + LOWEST LOSS (tiebreaker)", best5, display_cols)
    else:
        best5 = None
        print("\n  No combos with valid Sim RMSE for criterion 5.")

    # --- Top 10 table by capped loss ---
    print(f"\n{'='*70}")
    print(f"  TOP 10 COMBOS BY LOSS (capped > {args.loss_cap:.0f})")
    print(f"{'='*70}")
    if cap_mask.any():
        top10 = combined[cap_mask].nsmallest(10, "Mean_Loss")
        print(top10[display_cols].to_string(index=False))
    else:
        print("  None found.")

    # --- Save best_overall.json with all criteria ---
    best_overall = {}

    def row_to_hp_dict(row):
        """Extract hyperparameter dict from a result row."""
        return {
            "combo_index": int(row["Combo_Index"]),
            "degree": int(row["Degree"]),
            "tau": float(row["Tau"]),
            "lr": float(row["Lr"]),
            "n_tau": int(row["N_Tau"]),
            "measurement_noise": float(row["Measurement_Noise"]),
            "n_reparam_samples": int(row["N_Reparam_Samples"]),
            "mean_loss": float(row["Mean_Loss"]),
            "mean_rmse_omega": float(row["Mean_RMSE_Omega"]),
            "mean_sim_rmse_omega": float(row["Mean_Sim_RMSE_Omega"]) if pd.notna(row.get("Mean_Sim_RMSE_Omega")) else None,
            "num_success": int(row["Num_Success"]),
            "num_sim_success": int(row.get("Num_Sim_Success", 0)),
            "num_total": int(row["Num_Total"]),
        }

    if finite_mask.any():
        best_overall["best_loss_uncapped"] = row_to_hp_dict(best1)
    if cap_mask.any():
        best_overall["best_loss_capped"] = row_to_hp_dict(best2)
    if "Mean_Sim_RMSE_Omega" in combined.columns and sim_mask.any():
        best_overall["best_sim_rmse_uncapped"] = row_to_hp_dict(best3)
    if "Mean_Sim_RMSE_Omega" in combined.columns and both_mask.any():
        best_overall["best_sim_rmse_capped"] = row_to_hp_dict(best4)
    if best5 is not None:
        best_overall["best_sim_rmse_then_loss"] = row_to_hp_dict(best5)

    best_overall["metadata"] = {
        "run_name": args.run_name,
        "loss_cap": args.loss_cap,
        "total_combos": n_total,
        "combos_with_results": int(n_with_results),
        "combos_finite_loss": int(n_finite_loss),
        "combos_within_cap": int(n_capped),
    }

    best_json_path = os.path.join(results_dir, "best_overall.json")
    with open(best_json_path, "w") as f:
        json.dump(best_overall, f, indent=2)
    print(f"\nSaved best_overall.json (all criteria) to: {best_json_path}")

    print(f"Master CSV saved to: {master_csv}")


if __name__ == "__main__":
    main()
