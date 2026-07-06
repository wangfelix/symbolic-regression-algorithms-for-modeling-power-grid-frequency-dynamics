"""
SINDy Analysis: All 1-Hour Chunks from Synthetic Noisy Dataset

Fits a PySINDy model with best hyperparameters from tuning on each
1-hour chunk of the synthetic noisy dataset.
Designed to run as part of a SLURM array job.

Usage:
    python run_sindy_synthetic_all_chunks_1h.py --start-chunk 0 --end-chunk 99
    python run_sindy_synthetic_all_chunks_1h.py  # all chunks (single node)
"""
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pysindy as ps
import argparse
import csv
import json
import datetime

# =============================================================================
# Best Configuration (from HP tuning on synthetic_with_wiener, 1h chunks)
# =============================================================================
SIGMA = 15
DEGREE = 1
STLSQ_THRESHOLD = 1e-10
# Source: HP tuning run_SLURM_4011904, combo 45
# Mean RMSE omega=0.101723, Median=0.092908

CHUNK_SIZE = 3600  # 1 hour at 1s resolution


# =============================================================================
# Data Loading for Synthetic Data
# =============================================================================

def load_synthetic_data(data_path):
    """Load synthetic data and chunk into 1-hour windows."""
    print(f"Loading synthetic data from {data_path}...")

    with np.load(data_path) as data:
        omega = data['omega']
        theta = data['theta']

    print(f"  Total samples: {len(omega)}")

    n_chunks = len(omega) // CHUNK_SIZE
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunks.append({
            'omega': omega[start:end],
            'theta': theta[start:end],
            'chunk_index': i,
        })

    print(f"  Total 1-hour chunks: {len(chunks)}")
    print(f"  Keeping all {len(chunks)} chunks (noisy dataset has dynamics everywhere).")
    return chunks


def prepare_synthetic_chunk(chunk, dt=1.0, sigma=15):
    """Prepare a synthetic chunk for SINDy training."""
    omega_raw = chunk['omega'].copy()
    theta_raw = chunk['theta'].copy()

    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
        theta = np.cumsum(omega) * dt
    else:
        omega = omega_raw.copy()
        theta = theta_raw.copy()

    t = np.arange(len(omega)) * dt
    X = np.stack([theta, omega], axis=1)

    return t, X, omega_raw


# =============================================================================
# SINDy fitting for a single chunk
# =============================================================================

def fit_single_chunk(chunk):
    """Fit SINDy on one 1-hour chunk.

    Returns dict with RMSE, equations, and coefficients.
    """
    try:
        t_np, X_np, omega_raw = prepare_synthetic_chunk(chunk, dt=1.0, sigma=SIGMA)

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
        coeffs = model.coefficients()
        omega_coeffs = coeffs[1, :].tolist() if coeffs.shape[0] > 1 else [0.0] * 3

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
        "omega_coeffs": [np.nan] * 3,
        "failed": True,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SINDy analysis on all synthetic 1-hour chunks")
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int, default=-1)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../SVISE/synthetic_dataset_validation/synthetic_with_wiener.npz")

    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
        return

    all_chunks = load_synthetic_data(data_path)
    total_chunks = len(all_chunks)

    start = args.start_chunk
    end = args.end_chunk if args.end_chunk >= 0 else total_chunks - 1
    end = min(end, total_chunks - 1)

    if start > end or start >= total_chunks:
        print(f"Error: Invalid chunk range [{start}, {end}]. Total chunks: {total_chunks}")
        return

    n_chunks = end - start + 1

    print(f"\n{'=' * 60}")
    print(f"SINDy ANALYSIS — SYNTHETIC NOISY (1-HOUR CHUNKS)")
    print(f"{'=' * 60}")
    print(f"Sigma: {SIGMA}, Degree: {DEGREE}, STLSQ threshold: {STLSQ_THRESHOLD}")
    print(f"Total chunks in dataset: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"{'=' * 60}\n")

    # Output directory
    results_base = os.path.join(os.path.dirname(__file__), "results_sindy_synthetic_all_chunks_1h")
    if args.run_name:
        results_dir = os.path.join(results_base, args.run_name)
    else:
        results_dir = os.path.join(results_base, f"run_{timestamp}")

    os.makedirs(results_dir, exist_ok=True)

    # CSV header
    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    # Coefficient names depend on degree
    if DEGREE == 1:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega"]
    elif DEGREE == 2:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]
    else:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
                       "Coeff_Theta3", "Coeff_Theta2Omega", "Coeff_ThetaOmega2", "Coeff_Omega3"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Chunk_Index", "Original_Chunk_Index",
            "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
            "Eq_Theta", "Eq_Omega",
        ] + coeff_names)

    # Process chunks
    sim_rmse_list = []

    for i in range(start, end + 1):
        chunk = all_chunks[i]
        orig_idx = chunk['chunk_index']
        progress = i - start + 1

        if progress % 100 == 1 or progress == 1:
            print(f"\n--- Chunk {i} (orig #{orig_idx}) ({progress}/{n_chunks}) ---")

        result = fit_single_chunk(chunk)

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
                i, orig_idx,
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
            "dataset": "synthetic_with_wiener",
            "chunk_size": CHUNK_SIZE,
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
    print(f"SINDy ANALYSIS COMPLETE — Synthetic Noisy 1h [{start}, {end}]")
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
