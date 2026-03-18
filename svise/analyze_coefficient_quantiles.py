"""
Analyze quantiles of coefficient absolute values.

Calculates and prints percentiles (e.g., 25%, 50%, 75%, 90%) for each term's absolute values,
helping to determine a statistically sound cut-off threshold for noise vs signal.
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
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    all_coeffs = []
    for idx, row in df.iterrows():
        coeffs = parse_omega_equation(row.get("Eq_Omega", ""))
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
