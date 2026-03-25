"""
Compute 1D-L-KM reference model RMSE on empirical SK data.

For each chunk of empirical data, forward-simulates the 1D-L-KM Model 2 ODE
(using KM-estimated coefficients from the synthetic dataset generation) from
the chunk's initial conditions. Computes RMSE between simulated omega and
the Gaussian-filtered empirical omega.

This reproduces the 1D-L-KM reference column in Wen et al. Table 1 (Appendix A.2).

Usage:
    # 5-minute chunks (for comparison with SVISE 5-min results):
    python compute_1dlkm_reference_rmse.py --chunk-minutes 5 --sigma 60

    # 1-hour chunks (for comparison with Wen et al. Table 1):
    python compute_1dlkm_reference_rmse.py --chunk-minutes 60 --sigma 60

    # Process a subset of chunks (for SLURM array jobs):
    python compute_1dlkm_reference_rmse.py --start-chunk 0 --end-chunk 999
"""
import os
import sys
import argparse
import json
import csv
import datetime
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint


# =============================================================================
# Ground truth parameters (from SK KM estimation)
# =============================================================================
GROUND_TRUTH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "synthetic_dataset_validation", "ground_truth_params.json"
)

# Fallback values if JSON not found
DEFAULT_PARAMS = {
    "c_1": -0.009057647473133581,
    "c_2": -1.5317625512028024e-05,
    "Delta_P": 0.00554890484189234,
}

# RMSE threshold: intervals with RMSE >= this are counted as "unstable"
UNSTABLE_THRESHOLD = 0.5


# =============================================================================
# Data Loading
# =============================================================================
def load_data(data_path, limit_interpolation=10):
    """Load SK frequency data from parquet or pickle."""
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(
            method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(
            method='time', limit=limit_interpolation)

    return data


def get_valid_chunks(data, chunk_minutes=5):
    """Get all valid chunks of the specified duration.

    A valid chunk has exactly chunk_minutes*60 samples (1 Hz) with no missing data.
    """
    chunk_size = chunk_minutes * 60
    freq_str = f"{chunk_minutes}min"

    print(f"Finding all valid {chunk_minutes}-minute chunks ({chunk_size} samples each)...")

    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(
            subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor(freq_str))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == chunk_size:
            valid_chunks.append((chunk_start, group))

    if not valid_chunks:
        raise ValueError(f"No valid {chunk_minutes}-minute chunks found.")

    print(f"Found {len(valid_chunks)} valid {chunk_minutes}-minute chunks.")
    return valid_chunks


# =============================================================================
# 1D-L-KM Model 2 Forward Simulation
# =============================================================================
def model2_ode(state, t, c_omega, c_theta, P0_effective):
    """ODE system for 1D-L-KM Model 2.

    dtheta/dt = omega
    domega/dt = c_omega * omega + c_theta * theta + P0_effective

    P0_effective is the effective power mismatch for this chunk,
    computed as Delta_P * P(t) * sign(t) averaged or at the chunk's time.
    Since P0 varies within a chunk due to dispatch, we use a constant
    approximation for the chunk.
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = c_omega * omega + c_theta * theta + P0_effective
    return [dtheta_dt, domega_dt]


def simulate_chunk(omega_filtered, dt, c_omega, c_theta, P0_effective):
    """Forward-simulate Model 2 ODE on a single chunk.

    Args:
        omega_filtered: Gaussian-filtered empirical omega (rad/s), shape (N,)
        dt: time step (seconds)
        c_omega: damping coefficient (= c_1)
        c_theta: secondary control coefficient (= c_2 = c_1 * c_2_decay)
        P0_effective: effective power mismatch for this chunk

    Returns:
        sim_omega: simulated omega, shape (N,)  or None if simulation diverges
        rmse: RMSE between sim_omega and omega_filtered, or NaN if diverged
    """
    N = len(omega_filtered)
    t = np.arange(N) * dt

    # Initial conditions from filtered data
    theta0 = np.cumsum(omega_filtered[:1])[0] * dt  # ~0 for first point
    omega0 = omega_filtered[0]

    # Actually, theta is the integral of omega. For the initial condition,
    # theta(0) = 0 is a reasonable assumption (relative phase).
    theta0 = 0.0
    omega0 = omega_filtered[0]

    try:
        sol = odeint(model2_ode, [theta0, omega0], t,
                     args=(c_omega, c_theta, P0_effective),
                     full_output=False)
    except Exception:
        return None, np.nan

    sim_omega = sol[:, 1]

    # Check for divergence
    if not np.all(np.isfinite(sim_omega)):
        return None, np.nan

    # Divergence check: if max simulated value is > 100x max data value
    max_data = max(np.max(np.abs(omega_filtered)), 1e-10)
    if np.max(np.abs(sim_omega)) > 100 * max_data:
        return sim_omega, np.nan  # diverged

    rmse = np.sqrt(np.mean((sim_omega - omega_filtered) ** 2))
    return sim_omega, rmse


def estimate_P0_for_chunk(omega_filtered, c_omega, c_theta, dt):
    """Estimate the effective P0 for a chunk from the data itself.

    Since P0 = Delta_P * P(t) * sign(t) varies with dispatch,
    and we don't know the exact dispatch phase for an empirical chunk,
    we estimate P0 from the data using the mean residual:

        domega/dt ≈ c_omega * omega + c_theta * theta + P0
        => P0 ≈ mean(domega/dt - c_omega*omega - c_theta*theta)
    """
    theta = np.cumsum(omega_filtered) * dt
    domega_dt = np.gradient(omega_filtered, dt)

    residual = domega_dt - c_omega * omega_filtered - c_theta * theta
    P0_est = np.mean(residual)

    return P0_est


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute 1D-L-KM reference RMSE on empirical SK data")
    # Auto-detect data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(script_dir, "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(script_dir, "../dataset/Frequency_data_SK.pkl")
    if os.path.exists(parquet_path):
        default_data = parquet_path
    elif os.path.exists(pickle_path):
        default_data = pickle_path
    else:
        default_data = parquet_path  # will fail with clear error

    parser.add_argument("--data-path", type=str,
                        default=default_data,
                        help="Path to empirical dataset")
    parser.add_argument("--chunk-minutes", type=int, default=5,
                        choices=[5, 60],
                        help="Chunk duration in minutes (5 or 60)")
    parser.add_argument("--sigma", type=int, default=60,
                        help="Gaussian filter sigma (seconds). 0 = no filter.")
    parser.add_argument("--start-chunk", type=int, default=None,
                        help="First chunk index to process (for SLURM array)")
    parser.add_argument("--end-chunk", type=int, default=None,
                        help="Last chunk index to process (exclusive)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results_1dlkm_reference/)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for output subdirectory")
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"results_1dlkm_reference_{args.chunk_minutes}min")
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ground truth parameters
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, 'r') as f:
            gt = json.load(f)
        c_omega = gt["c_1"]
        c_theta = gt["c_2"]
        delta_P = gt["Delta_P"]
        print(f"Loaded ground truth: c_omega={c_omega:.6e}, c_theta={c_theta:.6e}, Delta_P={delta_P:.6e}")
    else:
        c_omega = DEFAULT_PARAMS["c_1"]
        c_theta = DEFAULT_PARAMS["c_2"]
        delta_P = DEFAULT_PARAMS["Delta_P"]
        print(f"Using default params: c_omega={c_omega:.6e}, c_theta={c_theta:.6e}, Delta_P={delta_P:.6e}")

    # Load data
    data = load_data(args.data_path)

    # Get valid chunks
    valid_chunks = get_valid_chunks(data, chunk_minutes=args.chunk_minutes)
    total_chunks = len(valid_chunks)

    # Determine chunk range
    start = args.start_chunk if args.start_chunk is not None else 0
    end = args.end_chunk if args.end_chunk is not None else total_chunks
    end = min(end, total_chunks)

    print(f"\nProcessing chunks {start} to {end-1} (of {total_chunks} total)")
    print(f"Chunk size: {args.chunk_minutes} min ({args.chunk_minutes * 60} samples)")
    print(f"Gaussian sigma: {args.sigma}")
    print(f"Output: {args.output_dir}")

    dt = 1.0  # 1 Hz sampling

    # CSV output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"chunks_{start:05d}_to_{end-1:05d}_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)

    csv_header = [
        "Chunk_Index", "Chunk_Start_Time",
        "RMSE_Omega", "Is_Unstable",
        "P0_Effective", "Omega_Std", "Omega_Mean"
    ]

    results = []
    n_processed = 0
    n_unstable = 0
    n_valid_rmse = 0
    rmse_values = []

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

        for i in range(start, end):
            chunk_start_time, chunk_df = valid_chunks[i]
            freq_values = chunk_df['freq'].values

            # Convert to omega (rad/s)
            if np.mean(freq_values) > 55:
                omega_raw = (freq_values - 60.0) * 2 * np.pi
            else:
                omega_raw = freq_values * 2 * np.pi

            # Gaussian filter
            if args.sigma > 0:
                omega_filtered = gaussian_filter1d(omega_raw, sigma=args.sigma)
            else:
                omega_filtered = omega_raw.copy()

            # Estimate effective P0 for this chunk from data
            P0_eff = estimate_P0_for_chunk(omega_filtered, c_omega, c_theta, dt)

            # Forward simulate
            sim_omega, rmse = simulate_chunk(
                omega_filtered, dt, c_omega, c_theta, P0_eff)

            # Classify stability
            is_unstable = False
            if np.isnan(rmse) or rmse >= UNSTABLE_THRESHOLD:
                is_unstable = True
                n_unstable += 1
            else:
                rmse_values.append(rmse)
                n_valid_rmse += 1

            n_processed += 1

            # Progress log
            rmse_str = f"{rmse:.6f}" if np.isfinite(rmse) else "DIVERGED"
            status = "UNSTABLE" if is_unstable else "OK"
            print(f"  [{n_processed}/{end-start}] Chunk {i} ({chunk_start_time}): "
                  f"RMSE={rmse_str} P0={P0_eff:.6e} [{status}]")

            writer.writerow([
                i, str(chunk_start_time),
                f"{rmse:.8f}" if np.isfinite(rmse) else "nan",
                int(is_unstable),
                f"{P0_eff:.8e}",
                f"{np.std(omega_filtered):.8e}",
                f"{np.mean(omega_filtered):.8e}",
            ])

    # Summary
    print(f"\n{'=' * 60}")
    print(f"1D-L-KM REFERENCE MODEL SUMMARY ({args.chunk_minutes}-min chunks)")
    print(f"{'=' * 60}")
    print(f"Total chunks processed: {n_processed}")
    print(f"Valid (RMSE < {UNSTABLE_THRESHOLD}): {n_valid_rmse}")
    print(f"Unstable: {n_unstable}")
    print(f"Share unstable: {n_unstable / max(n_processed, 1) * 100:.1f}%")

    if rmse_values:
        rmse_arr = np.array(rmse_values)
        print(f"\nRMSE (stable intervals only):")
        print(f"  Mean:   {rmse_arr.mean():.6f}")
        print(f"  Std:    {rmse_arr.std():.6f}")
        print(f"  Median: {np.median(rmse_arr):.6f}")
        print(f"  Min:    {rmse_arr.min():.6f}")
        print(f"  Max:    {rmse_arr.max():.6f}")

    # Save summary JSON
    summary = {
        "chunk_minutes": args.chunk_minutes,
        "sigma": args.sigma,
        "c_omega": c_omega,
        "c_theta": c_theta,
        "delta_P": delta_P,
        "total_chunks_processed": n_processed,
        "n_valid": n_valid_rmse,
        "n_unstable": n_unstable,
        "share_unstable": n_unstable / max(n_processed, 1),
        "rmse_threshold": UNSTABLE_THRESHOLD,
        "start_chunk": start,
        "end_chunk": end,
        "timestamp": timestamp,
    }
    if rmse_values:
        rmse_arr = np.array(rmse_values)
        summary["mean_rmse"] = float(rmse_arr.mean())
        summary["std_rmse"] = float(rmse_arr.std())
        summary["median_rmse"] = float(np.median(rmse_arr))

    summary_path = os.path.join(
        args.output_dir, f"summary_{start:05d}_to_{end-1:05d}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nCSV saved: {csv_path}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
