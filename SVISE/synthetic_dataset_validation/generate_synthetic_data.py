"""
Generate Synthetic Dataset using 1D-LK-M Model 2.

Estimates parameters (c_1, c_2_decay, Delta_P, epsilon) from real South Korean
power grid frequency data, then generates 30 days of noiseless synthetic data.

Usage:
    python generate_synthetic_data.py                    # Default sigma=60 (Wen's original)
    python generate_synthetic_data.py --sigma 15         # sigma=15 (PySR comparison)
    python generate_synthetic_data.py --sigma 0          # No de-trending
    python generate_synthetic_data.py --with-noise       # Also generate noisy variant
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

from utils import (
    data_cleaning, data_filter, power_mismatch, exp_decay,
    KM_Coeff_1, KM_Coeff_2, Euler_Maruyama_Model2
)


# =============================================================================
# Configuration
# =============================================================================
NOMINAL_FREQ = 60.0        # Hz (South Korea)
DISPATCH = 1               # Hourly dispatch (Korea's current schedule)
DELTA_T = 1.0              # Euler-Maruyama time step (seconds)
T_FINAL = 900 * 96 * 30    # 30 days in seconds (= 2,592,000)
SEED = 42
TREND = 1                  # Boolean-like: 1 = de-trend before KM, 0 = don't

# KM analysis parameters
BW_DRIFT = 0.1
BW_DIFF = 0.1
DIST_DRIFT = 500
DIST_DIFF = 500


# =============================================================================
# Data Loading
# =============================================================================

def load_sk_data(data_path):
    """Load South Korean frequency data."""
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    # Handle QI column if present
    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=10)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
        # Filter good quality data
        freq = data.loc[data['QI'] == 0, 'freq'].dropna()
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=10)
        freq = data['freq'].dropna()

    print(f"  Shape: {freq.shape}")
    print(f"  Time range: {freq.index.min()} to {freq.index.max()}")
    print(f"  Mean freq: {freq.mean():.4f} Hz")
    return freq


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from 1D-LK-M Model 2")
    parser.add_argument("--with-noise", action="store_true",
                        help="Also generate a noisy variant")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to SK frequency data pickle")
    parser.add_argument("--sigma", type=int, default=60,
                        help="Gaussian smoothing sigma for de-trending before KM estimation (default: 60). Use 0 for no de-trending.")
    args = parser.parse_args()

    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Try parquet first, then pickle
        parquet_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
        pickle_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.pkl")
        if os.path.exists(parquet_path):
            data_path = parquet_path
        elif os.path.exists(pickle_path):
            data_path = pickle_path
        else:
            print(f"Error: Data file not found. Tried:\n  {parquet_path}\n  {pickle_path}")
            sys.exit(1)

    # Output directory: use sigma suffix for non-default values
    script_dir_base = os.path.dirname(os.path.abspath(__file__))
    if args.sigma == 60:
        output_dir = script_dir_base  # backward compatible
    else:
        output_dir = os.path.join(script_dir_base, f"sigma_{args.sigma}")
        os.makedirs(output_dir, exist_ok=True)

    print(f"Sigma for de-trending: {args.sigma}")
    print(f"Output directory: {output_dir}")

    # ---- Step 1: Load and clean data ----
    freq = load_sk_data(data_path)

    # Apply data cleaning (adapted for 60 Hz grid)
    freq_clean = data_cleaning(freq, freq_limits=(59, 61))

    # Drop remaining NaN
    freq_clean = freq_clean.dropna()
    print(f"Clean data: {len(freq_clean)} samples")

    # Convert to angular velocity: omega = (freq - 60) * 2*pi
    omega = (freq_clean.values - NOMINAL_FREQ) * 2 * np.pi
    print(f"Angular velocity: mean={np.mean(omega):.6f}, std={np.std(omega):.6f} rad/s")

    # ---- Step 2: Estimate parameters ----
    print(f"\n{'=' * 60}")
    print("PARAMETER ESTIMATION")
    print(f"{'=' * 60}")

    # De-trend for KM analysis
    if args.sigma > 0:
        omega_filtered = data_filter(omega, sigma=args.sigma)
        omega_detrended = omega - TREND * omega_filtered
        print(f"  De-trending with sigma={args.sigma}")
    else:
        omega_detrended = omega.copy()
        print(f"  No de-trending (sigma=0)")

    # c_1: drift coefficient (primary control)
    print("\nEstimating c_1 (drift/damping)...")
    c_1 = KM_Coeff_1(omega_detrended, dim=1, time_res=1,
                      bandwidth=BW_DRIFT, dist=DIST_DRIFT, order=1)
    print(f"  c_1 = {c_1}")
    # c_1 is returned as array; for order=1 with ::2 slicing, it's the slope
    c_1_scalar = float(c_1[0]) if hasattr(c_1, '__len__') else float(c_1)

    # c_2_decay: secondary control (exponential decay rate)
    print("\nEstimating c_2_decay (secondary control)...")
    c_2_decay = exp_decay(omega, time_res=1, size=899)
    # If sigma=0 (no de-trending), disable secondary control coupling
    effective_trend = TREND if args.sigma > 0 else 0
    c_2_decay = effective_trend * c_2_decay
    c_2 = c_2_decay * c_1_scalar
    print(f"  c_2_decay = {c_2_decay:.8f}")
    print(f"  c_2 = c_2_decay * c_1 = {c_2:.8f}")

    # Delta_P: power mismatch
    print("\nEstimating Delta_P (power mismatch)...")
    Delta_P = power_mismatch(omega, avg_for_each_hour=False, dispatch=DISPATCH,
                             start_minute=0, end_minute=1/6,
                             length_seconds_of_interval=5)
    print(f"  Delta_P = {Delta_P:.8f}")

    # epsilon: noise amplitude (for reference, not used in noiseless)
    print("\nEstimating epsilon (noise amplitude)...")
    epsilon = KM_Coeff_2(omega_detrended, dim=1, time_res=1,
                         bandwidth=BW_DIFF, dist=DIST_DIFF,
                         multiplicative_noise=False)
    print(f"  epsilon = {epsilon:.8f}")

    # ---- Step 3: Print comparison with Wen's Balearic values ----
    print(f"\n{'=' * 60}")
    print("PARAMETER COMPARISON (SK vs Wen's Balearic)")
    print(f"{'=' * 60}")
    print(f"{'Parameter':<20} {'SK (estimated)':<20} {'Balearic (Wen)':<20}")
    print(f"{'-' * 60}")
    print(f"{'c_1':<20} {c_1_scalar:<20.6f} {-0.0295:<20.6f}")
    print(f"{'c_2_decay':<20} {c_2_decay:<20.8f} {'N/A':<20}")
    print(f"{'c_2 (c_2_decay*c_1)':<20} {c_2:<20.8f} {-4.52e-05:<20.8f}")
    print(f"{'Delta_P':<20} {Delta_P:<20.8f} {0.011:<20.8f}")
    print(f"{'epsilon':<20} {epsilon:<20.8f} {'(not reported)':<20}")
    print(f"{'=' * 60}")

    # ---- Step 4: Generate synthetic data ----
    print(f"\n{'=' * 60}")
    print("GENERATING SYNTHETIC DATA (Model 2, Noiseless)")
    print(f"{'=' * 60}")
    print(f"  Duration: {T_FINAL / 86400:.1f} days ({T_FINAL} seconds)")
    print(f"  Time step: {DELTA_T} s")
    print(f"  Dispatch: {DISPATCH} (hourly)")
    print(f"  Epsilon: 0 (noiseless)")

    omega_synth, theta_synth, P_driving = Euler_Maruyama_Model2(
        data=omega,
        c_1=c_1_scalar,
        c_2_decay=c_2_decay,
        Delta_P=Delta_P,
        epsilon=0,
        time_res=1,
        dispatch=DISPATCH,
        delta_t=DELTA_T,
        t_final=T_FINAL,
        seed=SEED,
    )

    print(f"  Generated {len(omega_synth)} samples")
    print(f"  omega range: [{np.min(omega_synth):.6f}, {np.max(omega_synth):.6f}] rad/s")
    print(f"  omega std: {np.std(omega_synth):.6f} rad/s")

    n_5min_chunks = len(omega_synth) // 300
    print(f"  Number of 5-min chunks (300 samples): {n_5min_chunks}")

    # ---- Step 5: Save outputs ----
    # Save synthetic data
    noiseless_path = os.path.join(output_dir, "synthetic_data_noiseless.pkl")
    synth_df = pd.DataFrame({
        'omega': omega_synth,
        'theta': theta_synth,
    })
    synth_df.to_pickle(noiseless_path)
    print(f"\nNoiseless data saved to: {noiseless_path}")

    # Save ground truth parameters
    params = {
        "c_1": c_1_scalar,
        "c_2_decay": c_2_decay,
        "c_2": c_2,
        "Delta_P": float(Delta_P),
        "epsilon": float(epsilon),
        "detrend_sigma": args.sigma,
        "dispatch": DISPATCH,
        "delta_t": DELTA_T,
        "t_final": T_FINAL,
        "seed": SEED,
        "nominal_freq_hz": NOMINAL_FREQ,
        "n_samples": len(omega_synth),
        "n_5min_chunks": n_5min_chunks,
        "model": "Model 2 (Linear with dispatch)",
        "equation": "domega/dt = c_1*omega + c_2_decay*c_1*theta + Delta_P*P(t)*sign(t)",
        "balearic_reference": {
            "c_1": -0.0295,
            "c_2": -4.52e-05,
            "Delta_P": 0.011,
        }
    }
    params_path = os.path.join(output_dir, "ground_truth_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Parameters saved to: {params_path}")

    # ---- Step 6: Generate sanity plots ----
    print("\nGenerating sanity plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: First 4 hours of omega
    n_4h = min(4 * 3600, len(omega_synth))
    t_hours = np.arange(n_4h) / 3600
    axes[0].plot(t_hours, omega_synth[:n_4h], linewidth=0.5)
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('omega (rad/s)')
    axes[0].set_title('Synthetic omega (noiseless) - First 4 hours')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: A few 5-min chunks
    for chunk_idx in [0, 10, 50, 100]:
        start = chunk_idx * 300
        end = start + 300
        if end <= len(omega_synth):
            t_chunk = np.arange(300)
            axes[1].plot(t_chunk, omega_synth[start:end], label=f'Chunk {chunk_idx}', alpha=0.7)
    axes[1].set_xlabel('Time within chunk (seconds)')
    axes[1].set_ylabel('omega (rad/s)')
    axes[1].set_title('Sample 5-minute chunks')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Driving force P(t)*sign(t)*Delta_P
    n_24h = min(24 * 3600, len(P_driving))
    t_hours_24 = np.arange(n_24h) / 3600
    axes[2].plot(t_hours_24, P_driving[:n_24h], linewidth=0.5)
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Driving force')
    axes[2].set_title('Power mismatch driving force - First 24 hours')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "synthetic_data_sanity_plots.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Sanity plots saved to: {plot_path}")

    # ---- Optional: Generate noisy variant ----
    if args.with_noise:
        print(f"\n{'=' * 60}")
        print("GENERATING SYNTHETIC DATA (Model 2, With Noise)")
        print(f"{'=' * 60}")

        omega_noisy, theta_noisy, _ = Euler_Maruyama_Model2(
            data=omega,
            c_1=c_1_scalar,
            c_2_decay=c_2_decay,
            Delta_P=Delta_P,
            epsilon=epsilon,
            time_res=1,
            dispatch=DISPATCH,
            delta_t=DELTA_T,
            t_final=T_FINAL,
            seed=SEED,
        )

        noisy_path = os.path.join(output_dir, "synthetic_data_noisy.pkl")
        noisy_df = pd.DataFrame({
            'omega': omega_noisy,
            'theta': theta_noisy,
        })
        noisy_df.to_pickle(noisy_path)
        print(f"Noisy data saved to: {noisy_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
