"""
Compute RMSE between synthetic and real datasets.

Compares Kramers-Moyal drift D1(omega) curves and omega distributions between
the synthetic noiseless dataset and the real South Korean frequency data.
This follows Wen et al. (2024) Appendix validation approach.

Usage:
    python compute_rmse_synthetic_vs_real.py
    python compute_rmse_synthetic_vs_real.py --data-path /path/to/real_data.pkl
"""
import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import data_cleaning, data_filter, KM_Coeff_1

try:
    from kramersmoyal import km
    HAS_KRAMERSMOYAL = True
except ImportError:
    HAS_KRAMERSMOYAL = False
    print("Warning: kramersmoyal not installed. Will use simplified KM estimation.")


def compute_km_drift(omega, bandwidth=0.1, dist=500, n_bins=200):
    """Compute Kramers-Moyal drift D1(omega) on a grid.

    Returns (omega_grid, D1_values) where D1(omega) = <delta_omega | omega> / delta_t
    """
    if HAS_KRAMERSMOYAL:
        # Use kramersmoyal library
        bins = np.array([n_bins])
        powers = np.array([[1]])  # D1 only
        kmc, edges = km(omega, bw=bandwidth, bins=bins, powers=powers)
        # kmc[0] = D0 (probability), kmc[1] = D1 (drift)
        if len(kmc) > 1:
            d1 = kmc[1]
            omega_grid = (edges[0][:-1] + edges[0][1:]) / 2
        else:
            d1 = kmc[0]
            omega_grid = (edges[0][:-1] + edges[0][1:]) / 2
        return omega_grid, d1
    else:
        # Simplified: bin omega, compute conditional mean of delta_omega
        omega_min, omega_max = np.percentile(omega, [1, 99])
        omega_grid = np.linspace(omega_min, omega_max, n_bins)
        bin_width = (omega_max - omega_min) / n_bins

        delta_omega = np.diff(omega)
        omega_centers = omega[:-1]

        d1 = np.full(n_bins, np.nan)
        for i in range(n_bins):
            lo = omega_grid[i] - bin_width / 2
            hi = omega_grid[i] + bin_width / 2
            mask = (omega_centers >= lo) & (omega_centers < hi)
            if mask.sum() >= dist:
                d1[i] = np.mean(delta_omega[mask])

        return omega_grid, d1


def load_real_data(data_path):
    """Load and clean real frequency data, return omega (rad/s)."""
    print(f"Loading real data from {data_path}...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=10)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
        freq = data.loc[data['QI'] == 0, 'freq'].dropna()
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=10)
        freq = data['freq'].dropna()

    freq_clean = data_cleaning(freq, freq_limits=(59, 61))
    freq_clean = freq_clean.dropna()

    omega = (freq_clean.values - 60.0) * 2 * np.pi
    print(f"  Real data: {len(omega)} samples, mean={np.mean(omega):.6f}, std={np.std(omega):.6f}")
    return omega


def load_synthetic_data(data_path):
    """Load synthetic omega data."""
    print(f"Loading synthetic data from {data_path}...")
    df = pd.read_pickle(data_path)
    omega = df['omega'].values
    print(f"  Synthetic data: {len(omega)} samples, mean={np.mean(omega):.6f}, std={np.std(omega):.6f}")
    return omega


def compute_rmse_drift(grid1, d1_1, grid2, d1_2):
    """Compute RMSE between two drift curves on their overlapping domain."""
    # Find overlapping range
    lo = max(grid1[~np.isnan(d1_1)].min(), grid2[~np.isnan(d1_2)].min())
    hi = min(grid1[~np.isnan(d1_1)].max(), grid2[~np.isnan(d1_2)].max())

    # Interpolate both to common grid
    n_pts = 200
    common_grid = np.linspace(lo, hi, n_pts)

    # Interpolate D1 values
    valid1 = ~np.isnan(d1_1)
    valid2 = ~np.isnan(d1_2)

    d1_interp1 = np.interp(common_grid, grid1[valid1], d1_1[valid1])
    d1_interp2 = np.interp(common_grid, grid2[valid2], d1_2[valid2])

    rmse = np.sqrt(np.mean((d1_interp1 - d1_interp2) ** 2))
    return rmse, common_grid, d1_interp1, d1_interp2


def main():
    parser = argparse.ArgumentParser(description="RMSE between synthetic and real data")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to real frequency data")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve real data path
    if args.data_path:
        real_path = args.data_path
    else:
        parquet_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
        pickle_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.pkl")
        if os.path.exists(parquet_path):
            real_path = parquet_path
        elif os.path.exists(pickle_path):
            real_path = pickle_path
        else:
            print(f"Error: Real data not found. Tried:\n  {parquet_path}\n  {pickle_path}")
            sys.exit(1)

    # Synthetic data path
    synth_path = os.path.join(script_dir, "synthetic_data_noiseless.pkl")
    if not os.path.exists(synth_path):
        print(f"Error: Synthetic data not found at {synth_path}")
        print("Run generate_synthetic_data.py first.")
        sys.exit(1)

    # Load data
    omega_real = load_real_data(real_path)
    omega_synth = load_synthetic_data(synth_path)

    # Use same length for fair comparison
    n_compare = min(len(omega_real), len(omega_synth))
    print(f"\nUsing {n_compare} samples for comparison")

    print(f"\n{'=' * 60}")
    print("COMPUTING KRAMERS-MOYAL DRIFT D1(omega)")
    print(f"{'=' * 60}")

    # De-trend before KM analysis (same as parameter estimation)
    omega_real_filtered = data_filter(omega_real[:n_compare], sigma=60)
    omega_real_detrended = omega_real[:n_compare] - omega_real_filtered

    omega_synth_filtered = data_filter(omega_synth[:n_compare], sigma=60)
    omega_synth_detrended = omega_synth[:n_compare] - omega_synth_filtered

    print("Computing KM drift for real data...")
    grid_real, d1_real = compute_km_drift(omega_real_detrended, bandwidth=0.1, dist=500)
    print("Computing KM drift for synthetic data...")
    grid_synth, d1_synth = compute_km_drift(omega_synth_detrended, bandwidth=0.1, dist=500)

    # RMSE between drift curves
    rmse_drift, common_grid, d1_r_interp, d1_s_interp = compute_rmse_drift(
        grid_real, d1_real, grid_synth, d1_synth
    )

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    # Distribution comparison
    print("\nOmega distribution comparison:")
    print(f"  Real:      mean={np.mean(omega_real[:n_compare]):.6f}, std={np.std(omega_real[:n_compare]):.6f}")
    print(f"  Synthetic: mean={np.mean(omega_synth[:n_compare]):.6f}, std={np.std(omega_synth[:n_compare]):.6f}")

    # Increment distribution
    inc_real = np.diff(omega_real[:n_compare])
    inc_synth = np.diff(omega_synth[:n_compare])
    print(f"\nIncrement distribution comparison:")
    print(f"  Real:      mean={np.mean(inc_real):.8f}, std={np.std(inc_real):.6f}")
    print(f"  Synthetic: mean={np.mean(inc_synth):.8f}, std={np.std(inc_synth):.6f}")

    # Direct RMSE of omega time series (limited use but reported for completeness)
    rmse_omega_direct = np.sqrt(np.mean((omega_real[:n_compare] - omega_synth[:n_compare]) ** 2))

    print(f"\nRMSE metrics:")
    print(f"  RMSE of KM drift D1(omega): {rmse_drift:.8f}")
    print(f"  RMSE of omega time series:  {rmse_omega_direct:.8f}")

    print(f"{'=' * 60}")

    # Save results
    results = {
        "n_samples_compared": n_compare,
        "rmse_km_drift_D1": float(rmse_drift),
        "rmse_omega_timeseries": float(rmse_omega_direct),
        "real_omega_stats": {
            "mean": float(np.mean(omega_real[:n_compare])),
            "std": float(np.std(omega_real[:n_compare])),
        },
        "synthetic_omega_stats": {
            "mean": float(np.mean(omega_synth[:n_compare])),
            "std": float(np.std(omega_synth[:n_compare])),
        },
        "real_increment_stats": {
            "mean": float(np.mean(inc_real)),
            "std": float(np.std(inc_real)),
        },
        "synthetic_increment_stats": {
            "mean": float(np.mean(inc_synth)),
            "std": float(np.std(inc_synth)),
        },
    }

    results_path = os.path.join(script_dir, "rmse_synthetic_vs_real.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_path}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: KM Drift comparison
    ax = axes[0, 0]
    valid_r = ~np.isnan(d1_real)
    valid_s = ~np.isnan(d1_synth)
    ax.plot(grid_real[valid_r], d1_real[valid_r], 'b-', alpha=0.7, label='Real (SK)')
    ax.plot(grid_synth[valid_s], d1_synth[valid_s], 'r--', alpha=0.7, label='Synthetic')
    ax.set_xlabel('omega (rad/s)')
    ax.set_ylabel('D1(omega)')
    ax.set_title(f'KM Drift D1(omega) — RMSE = {rmse_drift:.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Omega histograms
    ax = axes[0, 1]
    bins = np.linspace(
        min(np.percentile(omega_real[:n_compare], 1), np.percentile(omega_synth[:n_compare], 1)),
        max(np.percentile(omega_real[:n_compare], 99), np.percentile(omega_synth[:n_compare], 99)),
        100
    )
    ax.hist(omega_real[:n_compare], bins=bins, alpha=0.5, density=True, label='Real (SK)')
    ax.hist(omega_synth[:n_compare], bins=bins, alpha=0.5, density=True, label='Synthetic')
    ax.set_xlabel('omega (rad/s)')
    ax.set_ylabel('Density')
    ax.set_title('Omega Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Increment histograms
    ax = axes[1, 0]
    bins_inc = np.linspace(
        min(np.percentile(inc_real, 1), np.percentile(inc_synth, 1)),
        max(np.percentile(inc_real, 99), np.percentile(inc_synth, 99)),
        100
    )
    ax.hist(inc_real, bins=bins_inc, alpha=0.5, density=True, label='Real (SK)')
    ax.hist(inc_synth, bins=bins_inc, alpha=0.5, density=True, label='Synthetic')
    ax.set_xlabel('delta omega (rad/s)')
    ax.set_ylabel('Density')
    ax.set_title('Increment Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sample time series (first 1 hour)
    ax = axes[1, 1]
    n_1h = min(3600, n_compare)
    t_min = np.arange(n_1h) / 60.0
    ax.plot(t_min, omega_real[:n_1h], 'b-', alpha=0.5, linewidth=0.5, label='Real (SK)')
    ax.plot(t_min, omega_synth[:n_1h], 'r-', alpha=0.5, linewidth=0.5, label='Synthetic')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('omega (rad/s)')
    ax.set_title('Sample Time Series (first hour)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(script_dir, "rmse_synthetic_vs_real_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plots saved to: {plot_path}")


if __name__ == "__main__":
    main()
