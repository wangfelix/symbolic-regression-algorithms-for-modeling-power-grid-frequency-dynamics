"""
Plot the frequency of occurrence for each feature in the omega equations.

Iterates through all_chunks_combined.csv, parses the equations, and counts
how many times each coefficient's absolute value is >= the given threshold.
Outputs stats to CSV and generates a bar chart of the occurrence percentages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import os

matplotlib.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "figure.dpi": 150,
})

# --- Paths ---
SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_5min_all_chunks")
INPUT_CSV = os.path.join(RESULTS_DIR, "all_chunks_combined.csv")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "feature_occurrences.png")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "feature_occurrences_stats.csv")

# Cut-off for viewing a coefficient as "0"
THRESHOLD = 0.01

TERM_KEYS = [
    "const", "theta", "omega",
    "theta^2", "theta_omega", "omega^2",
    "theta^3", "theta^2_omega", "theta_omega^2", "omega^3",
]
TERM_LABELS = [
    "1", "θ", "ω",
    "θ²", "θω", "ω²",
    "θ³", "θ²ω", "θω²", "ω³",
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
            
    coeffs_array = np.array(all_coeffs)
    total_valid = len(coeffs_array)
    print(f"Valid equations: {total_valid}")

    if total_valid == 0:
        print("No valid equations found. Exiting.")
        return

    # Apply threshold to find non-zero occurrences
    abs_coeffs = np.abs(coeffs_array)
    is_active = abs_coeffs >= THRESHOLD
    
    counts = np.sum(is_active, axis=0)
    percentages = (counts / total_valid) * 100
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        "Feature": TERM_KEYS,
        "Label": TERM_LABELS,
        "Count": counts,
        "Percentage": percentages
    })
    stats_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved stats to: {OUTPUT_CSV}")
    
    print(f"\nFeature Occurrences (|coeff| >= {THRESHOLD}):")
    for i, key in enumerate(TERM_KEYS):
        print(f"{key:<15}: {counts[i]:>6} ({percentages[i]:>5.1f}%)")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(TERM_LABELS))
    bars = ax.bar(x, percentages, color="#2ca089", alpha=0.85, edgecolor="black", linewidth=0.8)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    ax.set_xticks(x)
    ax.set_xticklabels(TERM_LABELS)
    ax.set_xlabel("Feature Candidates")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title(f"Feature Selection Frequency (|coefficient| ≥ {THRESHOLD})")
    ax.set_ylim(0, max(percentages) + 15) # give room for labels
    
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
