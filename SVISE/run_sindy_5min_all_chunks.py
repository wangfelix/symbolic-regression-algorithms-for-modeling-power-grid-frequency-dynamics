"""
SINDy Analysis: All Valid 5-Minute Chunks

Fits a PySINDy model (PolynomialLibrary degree=2, STLSQ threshold=1e-10)
on each valid 5-min chunk with Gaussian smoothing sigma=15.
Designed to run as part of a SLURM array job.

Usage:
    python run_sindy_5min_all_chunks.py --start-chunk 0 --end-chunk 999
    python run_sindy_5min_all_chunks.py  # all chunks (single node)
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pysindy as ps
import argparse
import csv
import json
import datetime

# =============================================================================
# Configuration
# =============================================================================
SIGMA = 15
DEGREE = 2
STLSQ_THRESHOLD = 1e-10

# =============================================================================
# Data Loading (same as SVISE scripts)
# =============================================================================

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)

    return data


def get_all_valid_chunks(data):
    """Get ALL valid 5-minute chunks from the entire dataset."""
    print("Finding all valid 5-minute chunks in the dataset...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == 300:
            valid_chunks.append((chunk_start, group))

    if not valid_chunks:
        raise ValueError("No valid 5-minute chunks found in the dataset.")

    print(f"Found {len(valid_chunks)} valid 5-minute chunks in total.")
    return valid_chunks


def prepare_data(chunk_df, dt=1.0, sigma=15):
    freq_values = chunk_df['freq'].values

    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega = omega_raw

    theta = np.cumsum(omega) * dt
    t = np.arange(len(omega)) * dt
    X = np.stack([theta, omega], axis=1)

    return t, X, omega_raw


# =============================================================================
# SINDy fitting for a single chunk
# =============================================================================

def fit_single_chunk(chunk_df):
    """Fit SINDy on one 5-min chunk.

    Returns dict with RMSE, equations, and coefficients.
    """
    try:
        t_np, X_np, omega_raw = prepare_data(chunk_df, dt=1.0, sigma=SIGMA)

        # Dead chunk check
        if np.std(X_np[:, 1]) < 1e-8:
            return _failure_result("Dead chunk (omega std < 1e-8)")

        # SINDy model
        library = ps.PolynomialLibrary(degree=DEGREE)
        optimizer = ps.STLSQ(threshold=STLSQ_THRESHOLD)
        model = ps.SINDy(
            feature_names=["theta", "omega"],
            feature_library=library,
            optimizer=optimizer,
        )
        model.fit(X_np, t=1)

        # Extract equations
        eqs = model.equations()
        eq_theta_str = eqs[0] if len(eqs) > 0 else "N/A"
        eq_omega_str = eqs[1] if len(eqs) > 1 else "N/A"

        # Extract coefficients for omega equation
        # Library terms for degree=2, 2 features: [1, theta, omega, theta^2, theta*omega, omega^2]
        coeffs = model.coefficients()
        omega_coeffs = coeffs[1, :].tolist() if coeffs.shape[0] > 1 else [0.0] * 6

        # Forward simulate
        sim_rmse_omega = np.nan
        sim_rmse_theta = np.nan
        sim_rmse_total = np.nan
        try:
            sim = model.simulate(X_np[0], t_np, integrator="odeint")
            if np.any(np.isnan(sim)) or np.any(np.isinf(sim)):
                raise ValueError("Simulation diverged")
            if np.max(np.abs(sim[:, 1])) > 100 * np.max(np.abs(X_np[:, 1]) + 1e-10):
                raise ValueError("Simulation blew up")
            sim_rmse_omega = float(np.sqrt(np.mean((sim[:, 1] - X_np[:, 1])**2)))
            sim_rmse_theta = float(np.sqrt(np.mean((sim[:, 0] - X_np[:, 0])**2)))
            sim_rmse_total = float(np.sqrt((sim_rmse_omega**2 + sim_rmse_theta**2) / 2))
        except Exception as e:
            print(f"    Forward simulation failed: {e}")

        return {
            "sim_rmse_omega": sim_rmse_omega,
            "sim_rmse_theta": sim_rmse_theta,
            "sim_rmse_total": sim_rmse_total,
            "eq_theta": eq_theta_str,
            "eq_omega": eq_omega_str,
            "omega_coeffs": omega_coeffs,
            "failed": False,
        }

    except Exception as e:
        print(f"    Chunk fitting failed: {e}")
        return _failure_result(str(e))


def _failure_result(reason):
    return {
        "sim_rmse_omega": np.nan,
        "sim_rmse_theta": np.nan,
        "sim_rmse_total": np.nan,
        "eq_theta": f"FAILED: {reason}",
        "eq_omega": f"FAILED: {reason}",
        "omega_coeffs": [np.nan] * 6,
        "failed": True,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SINDy analysis on all valid 5-min chunks")
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int, default=-1)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load data
    parquet_path = os.path.join(os.path.dirname(__file__), "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
    if os.path.exists(parquet_path):
        DATA_PATH = parquet_path
    elif os.path.exists(pickle_path):
        DATA_PATH = pickle_path
    else:
        print(f"Error: Data file not found. Tried:\n  {parquet_path}\n  {pickle_path}")
        return

    data = load_data(DATA_PATH)
    all_chunks = get_all_valid_chunks(data)
    total_chunks = len(all_chunks)

    start = args.start_chunk
    end = args.end_chunk if args.end_chunk >= 0 else total_chunks - 1
    end = min(end, total_chunks - 1)

    if start > end or start >= total_chunks:
        print(f"Error: Invalid chunk range [{start}, {end}]. Total chunks: {total_chunks}")
        return

    n_chunks = end - start + 1

    print(f"\n{'=' * 60}")
    print(f"SINDy ANALYSIS — ALL CHUNKS")
    print(f"{'=' * 60}")
    print(f"Sigma: {SIGMA}, Degree: {DEGREE}, STLSQ threshold: {STLSQ_THRESHOLD}")
    print(f"Total valid chunks in dataset: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"{'=' * 60}\n")

    # Output directory
    results_base = os.path.join(os.path.dirname(__file__), "results_sindy_5min_all_chunks")
    if args.run_name:
        results_dir = os.path.join(results_base, args.run_name)
    else:
        results_dir = os.path.join(results_base, f"run_{timestamp}")

    os.makedirs(results_dir, exist_ok=True)

    # CSV header
    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                   "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Chunk_Index", "Chunk_Start_Time",
            "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
            "Eq_Theta", "Eq_Omega",
        ] + coeff_names)

    # Process chunks
    sim_rmse_list = []

    for i in range(start, end + 1):
        chunk_start_time, chunk_df = all_chunks[i]
        progress = i - start + 1

        if progress % 100 == 1 or progress == 1:
            print(f"\n--- Chunk {i} ({progress}/{n_chunks}) | Start: {chunk_start_time} ---")

        result = fit_single_chunk(chunk_df)

        sim_rmse_list.append(result["sim_rmse_omega"])

        if progress % 100 == 0 or progress == 1:
            print(f"    RMSE omega (Sim): {result['sim_rmse_omega']:.6f}" if not np.isnan(result['sim_rmse_omega']) else "    RMSE omega (Sim): nan")
            print(f"    Eq omega: {result['eq_omega']}")

        # Write to CSV
        coeffs = result["omega_coeffs"]
        coeff_strs = [f"{c:.10e}" if not np.isnan(c) else "nan" for c in coeffs]

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, str(chunk_start_time),
                f"{result['sim_rmse_omega']:.6f}" if not np.isnan(result['sim_rmse_omega']) else "nan",
                f"{result['sim_rmse_theta']:.6f}" if not np.isnan(result['sim_rmse_theta']) else "nan",
                f"{result['sim_rmse_total']:.6f}" if not np.isnan(result['sim_rmse_total']) else "nan",
                result['eq_theta'],
                result['eq_omega'],
            ] + coeff_strs)

        # Running stats every 500 chunks
        if progress % 500 == 0 or progress == n_chunks:
            valid_sim = [r for r in sim_rmse_list if not np.isnan(r)]
            mean_sim = np.mean(valid_sim) if valid_sim else float('nan')
            elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60
            rate = elapsed / progress
            eta = rate * (n_chunks - progress)
            print(f"\n  [Progress {progress}/{n_chunks}] "
                  f"Mean Sim RMSE omega: {mean_sim:.6f} | "
                  f"Valid: {len(valid_sim)}/{progress} | "
                  f"Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

    # Final summary
    valid_sim = [r for r in sim_rmse_list if not np.isnan(r)]

    summary = {
        "config": {
            "sigma": SIGMA,
            "degree": DEGREE,
            "stlsq_threshold": STLSQ_THRESHOLD,
            "features": ["theta", "omega"],
        },
        "chunk_range": [start, end],
        "n_chunks_processed": n_chunks,
        "n_chunks_succeeded_sim": len(valid_sim),
        "mean_sim_rmse_omega": float(np.mean(valid_sim)) if valid_sim else float('nan'),
        "std_sim_rmse_omega": float(np.std(valid_sim)) if valid_sim else float('nan'),
        "median_sim_rmse_omega": float(np.median(valid_sim)) if valid_sim else float('nan'),
        "timestamp": timestamp,
        "csv_file": csv_filename,
        "elapsed_minutes": (datetime.datetime.now() - start_time).total_seconds() / 60,
    }

    summary_path = os.path.join(results_dir, f"summary_{start:05d}_to_{end:05d}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"SINDy ANALYSIS COMPLETE — Chunks [{start}, {end}]")
    print(f"{'=' * 60}")
    print(f"Chunks processed: {n_chunks}")
    print(f"Chunks succeeded Sim: {len(valid_sim)} ({100*len(valid_sim)/n_chunks:.1f}%)")
    print(f"Mean Sim RMSE omega: {summary['mean_sim_rmse_omega']:.6f} ± {summary['std_sim_rmse_omega']:.6f}")
    print(f"Median Sim RMSE omega: {summary['median_sim_rmse_omega']:.6f}")
    print(f"Elapsed time: {summary['elapsed_minutes']:.1f} minutes")
    print(f"\nCSV: {csv_path}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
