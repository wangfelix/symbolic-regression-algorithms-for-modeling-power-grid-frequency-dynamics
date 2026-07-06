"""
Analyse omega equation coefficients from SINDy all_chunks_combined.csv.

Reads the explicit coefficient columns (Coeff_Const, Coeff_Theta, etc.)
and computes the mean absolute value and standard deviation across all
valid chunks.

Usage:
    python analyse_omega_coefficients_sindy.py --run-name run_SLURM_12345_sindy
"""
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, required=True)
args, _ = parser.parse_known_args()
RUN_NAME = args.run_name

INPUT_CSV = os.path.join(os.path.dirname(__file__),
                         "results_sindy_5min_all_chunks", RUN_NAME,
                         "all_chunks_combined.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),
                          "results_sindy_5min_all_chunks", RUN_NAME,
                          "omega_coefficient_stats.csv")

# Column name -> display name mapping
COEFF_COLUMNS = [
    ("Coeff_Const",       "const"),
    ("Coeff_Theta",       "theta"),
    ("Coeff_Omega",       "omega"),
    ("Coeff_Theta2",      "theta^2"),
    ("Coeff_ThetaOmega",  "theta_omega"),
    ("Coeff_Omega2",      "omega^2"),
    ("Coeff_Theta3",      "theta^3"),
    ("Coeff_Theta2Omega", "theta^2_omega"),
    ("Coeff_ThetaOmega2", "theta_omega^2"),
    ("Coeff_Omega3",      "omega^3"),
]


def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Convert coefficient columns to numeric
    for col, _ in COEFF_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to rows where coefficients are valid (not NaN, not from failed chunks)
    valid_mask = df["Sim_RMSE_Omega"].notna()
    for col, _ in COEFF_COLUMNS:
        if col in df.columns:
            valid_mask &= df[col].notna()

    valid = df[valid_mask]
    skipped = len(df) - len(valid)
    print(f"Valid equations: {len(valid)}")
    print(f"Skipped (NaN/invalid): {skipped}")

    if len(valid) == 0:
        print("No valid coefficients to analyse.")
        return

    # Compute mean and std of absolute coefficients
    header_parts = []
    values = []
    means = []
    stds = []

    for col, name in COEFF_COLUMNS:
        abs_vals = valid[col].abs()
        m = float(abs_vals.mean())
        s = float(abs_vals.std())
        means.append(m)
        stds.append(s)
        header_parts.extend([f"{name}_mean", f"{name}_std"])
        values.extend([m, s])

    # Write output CSV
    header = ",".join(header_parts)
    row_str = ",".join(f"{v:.6f}" for v in values)

    with open(OUTPUT_CSV, "w") as f:
        f.write(header + "\n")
        f.write(row_str + "\n")

    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"\nResults:")
    print(f"{'Term':<20} {'Mean':>12} {'Std':>12}")
    print("-" * 44)
    for i, (_, name) in enumerate(COEFF_COLUMNS):
        print(f"{name:<20} {means[i]:>12.6f} {stds[i]:>12.6f}")


if __name__ == "__main__":
    main()
