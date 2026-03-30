"""
Compare omega equation coefficients across multiple models.

Static script – edit the MODELS list below to add/remove models.
Each entry provides pre-computed mean/std values and a display label.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 18,
    "font.family": "sans-serif",
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
    "figure.dpi": 150,
    "axes.linewidth": 1.75,
})

# ---------------------------------------------------------------------------
# MODELS – add / remove entries here.
#   label : legend label for this model
#   mean  : list of coefficient means  (order: const, θ, ω, θ², θω, ω², θ³, θ²ω, θω², ω³)
#   std   : list of coefficient stds   (same order)
# ---------------------------------------------------------------------------
# MODELS = [
#     {
#         "label": "SVISE (C)",
#         "mean": [0.003086, 0.001150, 0.009498, 0.000210, 0.002509, 0.058910,
#                  0.000026, 0.000319, 0.010890, 0.177902],
#         "std":  [0.004430, 0.001849, 0.016051, 0.000496, 0.005826, 0.099131,
#                  0.000091, 0.001182, 0.036455, 0.337247],
#     },
#     {
#         "label": "SVISE (B)",
#         "mean": [0.000807, 0.000235, 0.000900, 0.000011, 0.000156, 0.003449,
#                  0.000000, 0.000000, 0.000000, 0.000000],
#         "std":  [0.001983, 0.000628, 0.005447, 0.000069, 0.001202, 0.017850,
#                  0.000000, 0.000000, 0.000000, 0.000000],
#     },
#     {
#         "label": "PySR",
#         "mean": [0.002391, 5.514e-04, 0.020120, 1.096e-04, 0.003126, 0.231659, 3.158e-05,
#                  6.534e-04, 0.046158, 1.530093],
#         "std":  [0.004482, 7.880e-04, 0.047122, 2.706e-04, 0.006726, 0.429718,
#                  1.261e-04, 0.001828, 0.135353 , 2.161165],
#     },
#     {
#         "label": "SINDy",
#         "mean": [0.004710 , 0.000527, 0.046212, 0.000043, 0.002573, 0.175667, 0,
#                  0, 0, 0],
#         "std":  [0.008832, 0.000617, 0.075763, 0.000100, 0.003356, 0.208575,
#                  0, 0, 0, 0],
#     },
# ]

MODELS = [
    {
        "label": "SVISE",
        "mean": [0.003086, 0.001150, 0.009498, 0.000210, 0.002509, 0.058910,
                 0.000026, 0.000319, 0.010890, 0.177902],
        "std":  [0.004430, 0.001849, 0.016051, 0.000496, 0.005826, 0.099131,
                 0.000091, 0.001182, 0.036455, 0.337247],
    },
    {
        "label": "PySR",
        "mean": [0.002391, 5.514e-04, 0.020120, 1.096e-04, 0.003126, 0.231659, 3.158e-05,
                 6.534e-04, 0.046158, 1.530093],
        "std":  [0.004482, 7.880e-04, 0.047122, 2.706e-04, 0.006726, 0.429718,
                 1.261e-04, 0.001828, 0.135353 , 2.161165],
    },
    {
        "label": "SINDy",
        "mean": [0.004710 , 0.000527, 0.046212, 0.000043, 0.002573, 0.175667, 0,
                 0, 0, 0],
        "std":  [0.008832, 0.000617, 0.075763, 0.000100, 0.003356, 0.208575,
                 0, 0, 0, 0],
    },
]

# ---------------------------------------------------------------------------
# Term labels (polynomial basis)
# ---------------------------------------------------------------------------
TERM_LABELS = [
    "1", "θ", "ω",
    "θ²", "θω", "ω²",
    "θ³", "θ²ω", "θω²", "ω³",
]

# Colour / marker palette – cycled automatically
_COLORS  = ["#2ca089", "#e07b39", "#6a5acd", "#d94f70", "#3b9dd4", "#8abe5a"]
_MARKERS = ["D", "o", "s", "^", "v", "P"]

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(SCRIPT_DIR, "omega_coefficients_comparison.png")


def plot_comparison(models, term_labels, output_path):
    """Plot coefficient means with std error bars for all models, log-scaled."""
    n_terms = len(term_labels)
    n_models = len(models)
    x = np.arange(n_terms)

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.20
    offsets = np.linspace(-(n_models - 1) * width / 2,
                           (n_models - 1) * width / 2,
                           n_models)

    for i, model in enumerate(models):
        means = np.array(model["mean"], dtype=float)
        stds  = np.array(model["std"],  dtype=float)
        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]

        # Mask out coefficients that the model does not include (mean=0 and std=0)
        present = ~((means == 0.0) & (stds == 0.0))
        pos = x[present] + offsets[i]
        m   = means[present]
        s   = stds[present]

        # Clamp lower error to avoid negative values on log scale
        lower_err = np.where(m - s > 0, s, m - 1e-10)
        lower_err = np.maximum(0, lower_err)
        upper_err = s

        # Guard zeros for log scale
        safe_m = np.maximum(1e-14, m)

        ax.errorbar(
            pos,
            safe_m,
            yerr=[lower_err, upper_err],
            fmt="none",
            ecolor=color,
            elinewidth=2.5,
            capsize=4,
            alpha=0.85,
        )
        ax.scatter(
            pos,
            safe_m,
            color=color,
            marker=marker,
            s=80,
            zorder=5,
            label=model["label"],
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(term_labels)
    ax.set_xlabel("Feature Candidates")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("Equation Coefficients – Model Comparison", pad=16)
    ax.legend(loc="upper center", ncol=n_models, frameon=True)
    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.4)
    ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    # Trim trailing all-zero cubic terms if every model has them as zero
    global TERM_LABELS
    terms = list(TERM_LABELS)
    models = MODELS

    # Check if all models have zeros for the cubic block (indices 6–9)
    all_cubic_zero = all(
        all(m["mean"][j] == 0.0 and m["std"][j] == 0.0 for j in range(6, 10))
        for m in models
    )
    if all_cubic_zero:
        terms = terms[:6]
        models = [
            {**m, "mean": m["mean"][:6], "std": m["std"][:6]}
            for m in models
        ]

    plot_comparison(models, terms, OUTPUT_PNG)


if __name__ == "__main__":
    main()
