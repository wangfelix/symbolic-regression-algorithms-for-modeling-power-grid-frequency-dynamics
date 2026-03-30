import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from datetime import datetime
import os


def _timestamp():
    """Return current timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_pdf_histogram_plot(df, plots_dir):
    """
    Create PDF histogram plots (linear and log scale)
    with KDE overlay, matching Oberhofer et al. style.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'freq' column.
    plots_dir : str
        Directory to save plots.
    """
    print("Creating PDF histogram plots...")

    freq_data = df['freq'].dropna()

    # Compute KDE once and reuse
    kde = gaussian_kde(freq_data)
    # Linear plot range
    x_lin = np.linspace(59.90, 60.10, 1000)
    # Log plot: KDE only within actual data range
    # (avoids astronomically small values in empty tails)
    f_min = freq_data.min()
    f_max = freq_data.max()
    x_log = np.linspace(f_min, f_max, 1000)

    plt.figure(figsize=(12, 6))

    # ---- Left: linear scale ----
    plt.subplot(1, 2, 1)
    plt.hist(
        freq_data, bins=100, density=True,
        alpha=0.7, color='skyblue', edgecolor='black'
    )
    plt.plot(
        x_lin, kde(x_lin),
        color='green', linewidth=1.5, label='Density'
    )
    plt.title(
        'Frequency Distribution (Histogram)',
        fontsize=16, fontweight='bold'
    )
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xlim(59.90, 60.10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Statistics annotation
    stats_text = (
        f"\u03bc={freq_data.mean():.4f} Hz, "
        f"\u03c3={freq_data.std():.4f} Hz\n"
        f"s={stats.skew(freq_data):.4f}, "
        f"\u03ba={stats.kurtosis(freq_data, fisher=False):.4f}"
    )
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat', alpha=0.8
        ),
        fontsize=18
    )

    # ---- Right: log scale (wider range) ----
    plt.subplot(1, 2, 2)
    plt.hist(
        freq_data, bins=150, density=True,
        alpha=0.7, color='lightcoral', edgecolor='black'
    )
    plt.plot(
        x_log, kde(x_log),
        color='green', linewidth=1.5, label='Density'
    )
    plt.yscale('log')
    plt.title(
        'Frequency Distribution (Log Scale)',
        fontsize=16, fontweight='bold'
    )
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Density (log scale)', fontsize=16)
    plt.xlim(f_min - 0.02, f_max + 0.02)
    plt.ylim(bottom=1e-4)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    fname = (
        f"frequency_pdf_histograms_{_timestamp()}.png"
    )
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PDF histogram plots saved to: {fpath}")

    # Print statistics
    print(f"\nFrequency statistics:")
    print(f"Mean: {freq_data.mean():.6f} Hz")
    print(f"Std: {freq_data.std():.6f} Hz")
    print(f"Skewness: {stats.skew(freq_data):.6f}")
    print(
        f"Kurtosis (Pearson): "
        f"{stats.kurtosis(freq_data, fisher=False):.6f}"
    )
    print(f"Min: {freq_data.min():.6f} Hz")
    print(f"Max: {freq_data.max():.6f} Hz")
    print(
        f"Range: "
        f"{freq_data.max() - freq_data.min():.6f} Hz"
    )

    return freq_data


def create_qq_box_plot(df, plots_dir):
    """
    Create Q-Q plot and Box plot.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'freq' column.
    plots_dir : str
        Directory to save plots.
    """
    print("Creating Q-Q and Box plots...")

    freq_data = df['freq'].dropna()

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    stats.probplot(freq_data, dist="norm", plot=plt)
    plt.title(
        'Q-Q Plot (Normality Check)',
        fontsize=16, fontweight='bold'
    )
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.boxplot(freq_data, vert=True)
    plt.title('Box Plot', fontsize=16, fontweight='bold')
    plt.ylabel('Frequency (Hz)', fontsize=16)
    plt.ylim(59.90, 60.10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"frequency_qq_box_plots_{_timestamp()}.png"
    fpath = os.path.join(plots_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Q-Q and Box plots saved to: {fpath}")


if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'dataset',
        'South_Korea_2024-08-15_2025-08-31_1s.pkl'
    )
    with open(dataset_path, 'rb') as f:
        df = pd.read_pickle(f)

    plots_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'plots'
    )
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    freq_data = create_pdf_histogram_plot(df, plots_dir)
    create_qq_box_plot(df, plots_dir)