import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
import os
import sys

# Add scripts directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acf_analysis import create_acf_plot
from pdf_analysis_split import (
    create_pdf_histogram_plot, create_qq_box_plot
)
from pdf_comparative_analysis import (
    create_comparative_pdf_plots
)


def _timestamp():
    """Return current timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_data():
    """Load the South Korean power grid frequency dataset"""
    print("Loading South Korean power grid frequency "
          "dataset...")
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'dataset',
        'South_Korea_2024-08-15_2025-08-31_1s.pkl'
    )
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f)
    return df


def dataset_summary(df):
    """
    Print a comprehensive dataset summary with all
    statistics needed for the paper's data section.
    """
    print("=" * 60)
    print("DATASET SUMMARY FOR PAPER")
    print("=" * 60)

    # ---- Basic structure ----
    print("\n--- Structure ---")
    print(f"Shape:   {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Index type: {type(df.index).__name__}")

    # ---- Time range ----
    print("\n--- Time Range ---")
    t_start = df.index.min()
    t_end = df.index.max()
    duration = t_end - t_start
    print(f"Start:    {t_start}")
    print(f"End:      {t_end}")
    print(f"Duration: {duration}")
    print(f"Duration: {duration.total_seconds():.0f} s "
          f"= {duration.total_seconds()/3600:.1f} h "
          f"= {duration.total_seconds()/86400:.1f} days")

    # ---- Sampling regularity ----
    print("\n--- Sampling ---")
    diffs = pd.Series(df.index).diff().dropna()
    print(f"Median dt: {diffs.median()}")
    print(f"Min dt:    {diffs.min()}")
    print(f"Max dt:    {diffs.max()}")
    mode_dt = diffs.mode()
    if len(mode_dt) > 0:
        print(f"Mode dt:   {mode_dt.iloc[0]}")

    # ---- Missing data ----
    print("\n--- Missing Data ---")
    print(f"Total rows:     {len(df):,}")
    for col in df.columns:
        n_null = df[col].isnull().sum()
        pct = 100 * n_null / len(df)
        print(f"  {col}: {n_null:,} NaN ({pct:.4f}%)")

    # Expected rows if perfectly continuous at 1 Hz
    expected_rows = int(duration.total_seconds()) + 1
    actual_rows = len(df)
    print(f"\nExpected rows (1 Hz, no gaps): {expected_rows:,}")
    print(f"Actual rows:                   {actual_rows:,}")
    print(f"Missing rows (gaps):           "
          f"{expected_rows - actual_rows:,}")
    if expected_rows > 0:
        coverage = 100 * actual_rows / expected_rows
        print(f"Coverage:                      {coverage:.2f}%")

    # ---- Frequency statistics ----
    freq_col = 'freq'
    if freq_col not in df.columns:
        # Try to find it
        for c in df.columns:
            if 'freq' in c.lower() or 'f' == c.lower():
                freq_col = c
                break

    if freq_col in df.columns:
        f = df[freq_col].dropna()
        print(f"\n--- Frequency Statistics ('{freq_col}', "
              f"N={len(f):,}) ---")
        print(f"Mean:     {f.mean():.6f} Hz")
        print(f"Std:      {f.std():.6f} Hz")
        print(f"Median:   {f.median():.6f} Hz")
        print(f"Min:      {f.min():.6f} Hz")
        print(f"Max:      {f.max():.6f} Hz")
        print(f"Range:    {f.max() - f.min():.6f} Hz")

        skew_val = stats.skew(f)
        # Pearson kurtosis (Gaussian = 3)
        kurt_pearson = stats.kurtosis(f, fisher=False)
        # Excess kurtosis (Gaussian = 0)
        kurt_excess = stats.kurtosis(f, fisher=True)
        print(f"Skewness:          {skew_val:.6f}")
        print(f"Kurtosis (Pearson, Gaussian=3): "
              f"{kurt_pearson:.6f}")
        print(f"Kurtosis (excess,  Gaussian=0): "
              f"{kurt_excess:.6f}")

        # Percentiles
        pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        print(f"\nPercentiles:")
        for p in pcts:
            print(f"  {p:5.1f}%: {np.percentile(f, p):.6f} Hz")
    else:
        print(f"\nWARNING: Could not find frequency column. "
              f"Available: {list(df.columns)}")

    # ---- Quality Indicator (if present) ----
    if 'QI' in df.columns:
        qi = df['QI']
        print(f"\n--- Quality Indicator ---")
        print(f"Unique values: {sorted(qi.dropna().unique())}")
        print(f"Value counts:\n{qi.value_counts().to_string()}")
        print(f"NaN count: {qi.isnull().sum()}")

    # ---- First and last rows ----
    print(f"\n--- First 3 rows ---")
    print(df.head(3).to_string())
    print(f"\n--- Last 3 rows ---")
    print(df.tail(3).to_string())

    print(f"\n{'=' * 60}")
    print("END DATASET SUMMARY")
    print("=" * 60)


def create_time_series_plot(df, plots_dir):
    """Create the main time series plot"""
    print("Creating main time series plot...")

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['freq'], linewidth=0.5, alpha=0.7)
    plt.title(
        'South Korean Power Grid Frequency',
        fontsize=14, fontweight='bold'
    )
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m-%d')
    )
    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    if 'QI' in df.columns:
        plt.subplot(2, 1, 2)
        plt.plot(
            df.index, df['QI'],
            linewidth=0.5, alpha=0.7, color='red'
        )
        plt.title(
            'Quality Indicator',
            fontsize=12, fontweight='bold'
        )
        plt.ylabel('QI', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        ax2 = plt.gca()
        ax2.xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m-%d')
        )
        ax2.xaxis.set_major_locator(
            mdates.MonthLocator()
        )

    plt.tight_layout()
    plot_path = os.path.join(
        plots_dir,
        f'south_korea_power_grid_frequency_{_timestamp()}.png'
    )
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def main():
    """Main analysis function"""
    print("South Korean Power Grid Frequency Analysis")
    print("=" * 50)

    # Load data
    df = load_data()

    # ---- Dataset summary (for paper) ----
    print("\n")
    dataset_summary(df)

    # Create plots directory
    plots_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'plots'
    )

    # Time series plot
    create_time_series_plot(df, plots_dir)

    # PDF analysis
    print("\n" + "=" * 50)
    print("PDF ANALYSIS")
    print("=" * 50)
    freq_data = create_pdf_histogram_plot(df, plots_dir)
    create_qq_box_plot(df, plots_dir)

    # Comparative PDF analysis
    print("\n" + "=" * 50)
    print("COMPARATIVE PDF ANALYSIS")
    print("=" * 50)
    create_comparative_pdf_plots(df, plots_dir)

    # ACF + Hurst analysis
    print("\n" + "=" * 50)
    print("ACF ANALYSIS")
    print("=" * 50)
    autocorr, lags = create_acf_plot(df, plots_dir)

    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"All plots saved to: {plots_dir}/")


if __name__ == "__main__":
    main()