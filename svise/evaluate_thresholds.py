"""
Analyze different thresholds for feature selection.

Tests various threshold values to see how the feature selection percentages change,
which helps determine a mathematically sound cut-off point for numerical noise.
"""

import pandas as pd
import numpy as np
import re
import os

INPUT_CSV = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", "all_chunks_combined.csv")

TERM_KEYS = [
    "const", "theta", "omega",
    "theta^2", "theta_omega", "omega^2",
    "theta^3", "theta^2_omega", "theta_omega^2", "omega^3",
]

def parse_omega_equation(eq_str):
    if not isinstance(eq_str, str) or "nan" in eq_str.lower():
        return None
    parts = eq_str.split(" + ")
    if len(parts) != 10:
        return None
    
    coefficients = []
    for part in parts:
        part = part.strip()
        match = re.match(r'^([+-]?\d+\.?\d*)', part)
        if match:
            coefficients.append(float(match.group(1)))
        else:
            return None
    return coefficients

def main():
    df = pd.read_csv(INPUT_CSV)
    all_coeffs = [parse_omega_equation(row.get("Eq_Omega", "")) for _, row in df.iterrows()]
    all_coeffs = [c for c in all_coeffs if c is not None]
    
    abs_coeffs = np.abs(np.array(all_coeffs))
    total = len(abs_coeffs)
    
    thresholds_to_test = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    print(f"{'Threshold':<10} | " + " | ".join([f"{k[:7]:<7}" for k in TERM_KEYS]))
    print("-" * 110)
    
    for th in thresholds_to_test:
        is_active = abs_coeffs >= th
        percentages = (np.sum(is_active, axis=0) / total) * 100
        
        row_str = f"{th:<10.3f} | " + " | ".join([f"{p:>6.1f}%" for p in percentages])
        print(row_str)

        
if __name__ == "__main__":
    main()
