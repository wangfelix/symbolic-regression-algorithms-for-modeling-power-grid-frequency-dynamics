"""
Analyse omega equation coefficients from all_chunks_combined.csv.

Parses the Eq_Omega column, extracts coefficients for each polynomial term,
and computes the mean and standard deviation across all valid (non-NaN) chunks.

Output: omega_coefficient_stats.csv
"""

import pandas as pd
import numpy as np
import re
import os

# --- Paths ---
INPUT_CSV = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", "all_chunks_combined.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", "omega_coefficient_stats.csv")

# --- Term order (matches the equation string order) ---
TERM_NAMES = [
    "const",        # e.g. -0.059691
    "theta",        # e.g. -0.05687theta
    "omega",        # e.g. -0.01368omega
    "theta^2",      # e.g. -0.09118theta^2
    "theta_omega",  # e.g.  0.05395theta omega  (= theta*omega)
    "omega^2",      # e.g.  0.00926omega^2
    "theta^3",      # e.g. -0.01502theta^3
    "theta^2_omega",# e.g.  0.00934theta^2 omega
    "theta_omega^2",# e.g.  0.00506theta omega^2
    "omega^3",      # e.g. -0.05837omega^3
]


def parse_omega_equation(eq_str):
    """
    Parse an Eq_Omega string and return a list of 10 coefficients
    in the order: const, theta, omega, theta^2, theta*omega, omega^2,
                  theta^3, theta^2*omega, theta*omega^2, omega^3.

    Returns None if the equation contains NaN coefficients.
    """
    if not isinstance(eq_str, str):
        return None

    # Check for NaN equations
    if "nan" in eq_str.lower():
        return None

    # The equation format is:
    #   coeff1 + coeff2*theta + coeff3*omega + coeff4*theta^2 + ...
    # Split by " + " to get individual terms
    # Note: coefficients can be negative, appearing as "+ -0.05687theta"
    parts = eq_str.split(" + ")

    if len(parts) != 10:
        return None

    coefficients = []
    for part in parts:
        part = part.strip()
        # Extract the numeric coefficient from the term
        # The coefficient is at the beginning of each term, before any variable name
        # Use regex to extract the leading number (including sign and decimal)
        match = re.match(r'^([+-]?\d+\.?\d*)', part)
        if match:
            coefficients.append(float(match.group(1)))
        else:
            return None

    return coefficients


def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Parse all omega equations
    all_coeffs = []
    skipped = 0
    for idx, row in df.iterrows():
        coeffs = parse_omega_equation(row["Eq_Omega"])
        if coeffs is not None:
            all_coeffs.append(coeffs)
        else:
            skipped += 1

    print(f"Valid equations: {len(all_coeffs)}")
    print(f"Skipped (NaN/invalid): {skipped}")

    # Convert to numpy array for easy statistics
    coeffs_array = np.array(all_coeffs)

    # Use absolute values for feature importance analysis
    coeffs_abs = np.abs(coeffs_array)

    # Compute mean and std of absolute coefficients for each term
    means = np.mean(coeffs_abs, axis=0)
    stds = np.std(coeffs_abs, axis=0)

    # Build the output CSV header and row
    header_parts = []
    values = []
    for i, name in enumerate(TERM_NAMES):
        header_parts.append(f"{name}_mean")
        header_parts.append(f"{name}_std")
        values.append(means[i])
        values.append(stds[i])

    header = ",".join(header_parts)
    row_str = ",".join(f"{v:.6f}" for v in values)

    # Write output
    with open(OUTPUT_CSV, "w") as f:
        f.write(header + "\n")
        f.write(row_str + "\n")

    print(f"\nOutput saved to: {OUTPUT_CSV}")
    print(f"\nResults:")
    print(f"{'Term':<15} {'Mean':>12} {'Std':>12}")
    print("-" * 39)
    for i, name in enumerate(TERM_NAMES):
        print(f"{name:<15} {means[i]:>12.6f} {stds[i]:>12.6f}")


if __name__ == "__main__":
    main()
