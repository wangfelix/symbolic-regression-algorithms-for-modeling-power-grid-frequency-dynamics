"""
Analyze quantiles of coefficient absolute values.

Calculates and prints percentiles (e.g., 25%, 50%, 75%, 90%) for each term's absolute values,
helping to determine a statistically sound cut-off threshold for noise vs signal.
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", RUN_NAME)
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks")

INPUT_CSV = os.path.join(RESULTS_DIR, "all_chunks_combined.csv")

TERM_KEYS = [
    "const", "theta", "omega",
    "theta^2", "theta_omega", "omega^2",
    "theta^3", "theta^2_omega", "theta_omega^2", "omega^3",
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
    
    all_coeffs = []
    for idx, row in df.iterrows():
        coeffs = parse_omega_equation(row.get("Eq_Omega_Physical", row.get("Eq_Omega", "")))
        if coeffs is not None:
            all_coeffs.append(coeffs)
            
    abs_coeffs = np.abs(np.array(all_coeffs))
    
    quantiles_to_check = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    
    print("\nQuantiles of absolute coefficients:")
    print(f"{'Term':<15} | " + " | ".join([f"{int(q*100):>3}th %" for q in quantiles_to_check]))
    print("-" * 75)
    
    for i, key in enumerate(TERM_KEYS):
        term_data = abs_coeffs[:, i]
        quantiles = np.quantile(term_data, quantiles_to_check)
        row_str = f"{key:<15} | " + " | ".join([f"{val:>7.4f}" for val in quantiles])
        print(row_str)

if __name__ == "__main__":
    main()
