"""
SINDy Analysis: All Active 1-Hour Chunks from Synthetic Noiseless Data.

Fits a PySINDy model (PolynomialLibrary degree=2, STLSQ threshold=1e-10)
on each active 1-hour chunk from the synthetic noiseless dataset.
No smoothing is applied since the data is noiseless.

Designed for SLURM array jobs.

Usage:
    python run_sindy_synthetic_noiseless_1h.py --start-chunk 0 --end-chunk 99
    python run_sindy_synthetic_noiseless_1h.py  # all chunks
"""
import os
import sys
import pandas as pd
import numpy as np
import pysindy as ps
import argparse
import csv
import json
import datetime

# =============================================================================
# Configuration
# =============================================================================
DEGREE = 1
STLSQ_THRESHOLD = 1e-10

# Chunk configuration (same as SVISE 1h script)
CHUNK_SIZE = 3600        # 1 hour at 1s resolution

# Filter threshold for dead chunks
MIN_OMEGA_STD = 1e-4


# =============================================================================
# Data Loading for Synthetic Data (1-hour chunks)
# =============================================================================

def load_synthetic_data(data_path):
    """Load synthetic data and chunk into 1-hour windows, filtering dead chunks."""
    print(f"Loading synthetic data from {data_path}...")
    df = pd.read_pickle(data_path)
    omega = df['omega'].values
    theta = df['theta'].values
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

    active_chunks = [c for c in chunks if np.std(c['omega']) >= MIN_OMEGA_STD]
    print(f"  Active chunks (omega std >= {MIN_OMEGA_STD}): {len(active_chunks)}/{len(chunks)}")
    return active_chunks


def prepare_synthetic_chunk(chunk, dt=1.0):
    """Prepare a synthetic chunk for SINDy fitting. No smoothing."""
    omega = chunk['omega'].copy()
    theta = chunk['theta'].copy()

    t = np.arange(len(omega)) * dt
    X = np.stack([theta, omega], axis=1)

    return t, X


# =============================================================================
# SINDy fitting for a single chunk
# =============================================================================

def fit_single_chunk(chunk):
    """Fit SINDy on one 1-hour synthetic chunk.

    Returns dict with RMSE, equations, and coefficients.
    """
    try:
        t_np, X_np = prepare_synthetic_chunk(chunk, dt=1.0)

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
    parser = argparse.ArgumentParser(description="SINDy analysis on all active synthetic 1-hour chunks")
    parser.add_argument("--start-chunk", type=int, default=0,
                        help="Start chunk index (inclusive, 0-indexed into active chunks)")
    parser.add_argument("--end-chunk", type=int, default=-1,
                        help="End chunk index (inclusive). -1 = last chunk.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for the run folder")
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load synthetic data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "synthetic_data_noiseless.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
        print("Run generate_synthetic_data.py first.")
        sys.exit(1)

    all_chunks = load_synthetic_data(data_path)
    total_chunks = len(all_chunks)

    # Determine chunk range
    start = args.start_chunk
    end = args.end_chunk if args.end_chunk >= 0 else total_chunks - 1
    end = min(end, total_chunks - 1)

    if start > end or start >= total_chunks:
        print(f"Error: Invalid chunk range [{start}, {end}]. Total active chunks: {total_chunks}")
        return

    n_chunks = end - start + 1

    print(f"\n{'=' * 60}")
    print(f"SINDy ANALYSIS — SYNTHETIC NOISELESS (1-HOUR CHUNKS)")
    print(f"{'=' * 60}")
    print(f"Degree: {DEGREE}, STLSQ threshold: {STLSQ_THRESHOLD}")
    print(f"No smoothing (noiseless data)")
    print(f"Chunk size: {CHUNK_SIZE} samples (1 hour)")
    print(f"Total active chunks: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"{'=' * 60}\n")

    # Output directory
    results_base = os.path.join(script_dir, "results_sindy_synthetic_noiseless_1h")
    if args.run_name:
        results_dir = os.path.join(results_base, args.run_name)
    else:
        results_dir = os.path.join(results_base, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # CSV header
    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega"]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Active_Chunk_Index", "Original_Chunk_Index",
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
            print(f"\n--- Active chunk {i} (orig #{orig_idx}) ({progress}/{n_chunks}) ---")

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

        # Running stats every 100 chunks
        if progress % 100 == 0 or progress == n_chunks:
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
            "degree": DEGREE,
            "stlsq_threshold": STLSQ_THRESHOLD,
            "smoothing": "none (noiseless data)",
            "features": ["theta", "omega"],
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
    print(f"SINDy ANALYSIS COMPLETE — Synthetic Noiseless 1h [{start}, {end}]")
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
