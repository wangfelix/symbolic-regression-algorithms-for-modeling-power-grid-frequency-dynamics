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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, default="run_SLURM_3708675")
args, _ = parser.parse_known_args()
RUN_NAME = args.run_name

# --- Paths ---
INPUT_CSV = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", RUN_NAME, "all_chunks_combined.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", RUN_NAME, "omega_coefficient_stats.csv")

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


import functools

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    _SYMPY_AVAILABLE = True
    _SYMPY_TRANSFORMS = (standard_transformations + (implicit_multiplication_application, convert_xor))
    _THETA, _OMEGA = sympy.symbols('theta omega')
    _X0, _X1 = sympy.symbols('x0 x1')
    _GLOBAL_DICT = {
        'theta': _THETA, 'omega': _OMEGA, 'x0': _X0, 'x1': _X1,
        'Symbol': sympy.Symbol, 'Float': sympy.Float, 'Integer': sympy.Integer,
        'Add': sympy.Add, 'Mul': sympy.Mul, 'Pow': sympy.Pow,
    }
except ImportError:
    _SYMPY_AVAILABLE = False

@functools.lru_cache(maxsize=100000)
def parse_omega_equation(eq_str):
    if not _SYMPY_AVAILABLE:
        print("Please install sympy: pip install sympy")
        return None

    if not isinstance(eq_str, str) or "nan" in eq_str.lower() or "bounds" in eq_str.lower() or "error" in eq_str.lower() or eq_str == "N/A":
        return None

    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS, global_dict=_GLOBAL_DICT)
        expr = sympy.expand(expr)
    except Exception:
        return None
    
    # Safely evaluate both syntax versions (physics 'theta' vs abstract 'x0') natively
    try:
        # Standardize strictly to theta and omega internally before extracting coefficients
        expr = expr.subs({_X0: _THETA, _X1: _OMEGA})
        
        c0 = float(expr.subs({_THETA: 0, _OMEGA: 0}))
        c1 = float(expr.coeff(_THETA, 1).subs({_OMEGA: 0}))
        c2 = float(expr.coeff(_OMEGA, 1).subs({_THETA: 0}))
        c3 = float(expr.coeff(_THETA, 2).subs({_OMEGA: 0}))
        c4 = float(expr.coeff(_THETA*_OMEGA))
        c5 = float(expr.coeff(_OMEGA, 2).subs({_THETA: 0}))
        c6 = float(expr.coeff(_THETA, 3).subs({_OMEGA: 0}))
        c7 = float(expr.coeff(_THETA**2 * _OMEGA))
        c8 = float(expr.coeff(_THETA * _OMEGA**2))
        c9 = float(expr.coeff(_OMEGA, 3).subs({_THETA: 0}))
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    except Exception:
        return None


def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows: {len(df)}")

    # Parse all omega equations
    all_coeffs = []
    skipped = 0
    for idx, row in df.iterrows():
        # Point parser at the final mathematically transformed physical equations
        coeffs = parse_omega_equation(row.get("Eq_Omega_Physical", row.get("Eq_Omega", "")))
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
