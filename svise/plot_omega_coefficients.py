"""
Plot omega equation coefficients with error bars on a logarithmic scale.

Reads omega_coefficient_stats.csv and produces a feature importance plot
with mean (marker) and standard deviation (error bar) for each polynomial term.

Designed to later support multiple models side-by-side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})

import sys
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "run_SLURM_3708675"

# --- Paths ---
SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_5min_all_chunks", RUN_NAME)
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_5min_all_chunks")

INPUT_CSV = os.path.join(RESULTS_DIR, "omega_coefficient_stats.csv")
OUTPUT_PNG_LOG = os.path.join(RESULTS_DIR, "omega_coefficient_importance_log.png")
OUTPUT_PNG_LIN = os.path.join(RESULTS_DIR, "omega_coefficient_importance_linear.png")

# --- Term display labels (using Unicode for Greek letters) ---
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


def load_stats(csv_path):
    """Load mean/std from a coefficient stats CSV file."""
    df = pd.read_csv(csv_path)
    means = []
    stds = []
    for key in TERM_KEYS:
        means.append(df[f"{key}_mean"].values[0])
        stds.append(df[f"{key}_std"].values[0])
    return np.array(means), np.array(stds)


def plot_coefficients(models, output_path, use_log=True):
    """
    Plot coefficient importance for one or more models.

    Parameters
    ----------
    models : list of dict
        Each dict has keys: 'name', 'means', 'stds', 'color', 'marker'
    output_path : str
        Path to save the figure.
    use_log : bool
        Whether to use logarithmic scale
    """
    n_terms = len(TERM_LABELS)
    n_models = len(models)
    x = np.arange(n_terms)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Offset positions when multiple models are plotted
    width = 0.25
    offsets = np.linspace(-(n_models - 1) * width / 2,
                           (n_models - 1) * width / 2,
                           n_models)

    for i, model in enumerate(models):
        means = model["means"]
        stds = model["stds"]

        if use_log:
            # Clamp lower error bar so it doesn't go below zero on log scale
            # Also mathematically absolutely guard against negative error lengths if 'means' identically equals 0
            lower_err = np.where(means - stds > 0, stds, means - 1e-10)
            lower_err = np.maximum(0, lower_err)
            upper_err = stds
            
            # Guard the scatter plots from attempting to draw absolute zero on Log scales safely
            safe_means = np.maximum(1e-14, means)
        else:
            lower_err = stds
            upper_err = stds
            safe_means = means

        ax.errorbar(
            x + offsets[i],
            safe_means,
            yerr=[lower_err, upper_err],
            fmt="none",
            ecolor=model["color"],
            elinewidth=1.8,
            capsize=0,
            alpha=0.85,
        )
        ax.scatter(
            x + offsets[i],
            safe_means,
            color=model["color"],
            marker=model["marker"],
            s=80,
            zorder=5,
            label=model["name"],
            edgecolors="white",
            linewidths=0.5,
        )

    if use_log:
        ax.set_yscale("log")
        
    ax.set_xticks(x)
    ax.set_xticklabels(TERM_LABELS)
    ax.set_xlabel("Feature Candidates")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Coefficients with Error Bars")
    ax.legend(loc="upper right")
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    # --- Load SVISE model stats ---
    means, stds = load_stats(INPUT_CSV)

    global TERM_KEYS, TERM_LABELS
    # Dynamically strip identical 0.0 cubic terms if strictly quadratic equations arose
    if np.all(means[6:] == 0.0) and np.all(stds[6:] == 0.0):
        means = means[:6]
        stds = stds[:6]
        TERM_KEYS = TERM_KEYS[:6]
        TERM_LABELS = TERM_LABELS[:6]

    models = [
        {
            "name": "SVISE",
            "means": means,
            "stds": stds,
            "color": "#2ca089",   # teal green
            "marker": "D",        # diamond
        },
    ]

    plot_coefficients(models, OUTPUT_PNG_LOG, use_log=True)
    plot_coefficients(models, OUTPUT_PNG_LIN, use_log=False)


if __name__ == "__main__":
    main()
