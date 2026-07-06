"""
SINDy Hyperparameter Tuning on Synthetic Noisy Dataset (1-Hour Chunks)

Exhaustive grid search over sigma, degree, and STLSQ threshold using
the synthetic dataset with wiener noise in 1-hour chunks.

Usage:
    python run_sindy_synthetic_hp_tuning_1h.py --combo-index 0
    python run_sindy_synthetic_hp_tuning_1h.py --combo-index 0 --run-name run_SLURM_12345
    python run_sindy_synthetic_hp_tuning_1h.py  # evaluate all combos sequentially
    python run_sindy_synthetic_hp_tuning_1h.py --dry-run
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
import itertools
import random

# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================
HYPERPARAMETER_SPACE = {
    "sigma": [0, 5, 10, 15],
    "degree": [1, 2, 3],
    "threshold": [1e-10, 1e-5, 1e-3, 1e-2, 1e-1],
}

# Total: 4 * 3 * 5 = 60 combinations


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

    # Chunk into 1-hour windows (3600 samples at 1s)
    chunk_size = 3600
    n_chunks = len(omega) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append({
            'omega': omega[start:end],
            'theta': theta[start:end],
        })

    print(f"  Total 1-hour chunks: {len(chunks)}")
    print(f"  Keeping all {len(chunks)} chunks without filtering.")
        
    return chunks


def prepare_synthetic_chunk(chunk, dt=1.0, sigma=0):
    """Prepare a synthetic chunk for SINDy training."""
    omega_raw = chunk['omega'].copy()
    theta_raw = chunk['theta'].copy()

    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
        # Recalculate theta from filtered omega
        theta = np.cumsum(omega) * dt
    else:
        omega = omega_raw.copy()
        # Use E-M theta directly
        theta = theta_raw.copy()

    t = np.arange(len(omega)) * dt
    X = np.stack([theta, omega], axis=1)

    return t, X, omega_raw


# =============================================================================
# SINDy fitting for a single chunk
# =============================================================================

def fit_single_chunk(chunk, hyperparams):
    """Fit SINDy on one 1-hour chunk with given hyperparameters.

    Returns (sim_rmse_omega, sim_rmse_theta, sim_rmse_total) or NaNs on failure.
    """
    sigma = hyperparams["sigma"]
    degree = hyperparams["degree"]
    threshold = hyperparams["threshold"]

    try:
        t_np, X_np, omega_raw = prepare_synthetic_chunk(chunk, dt=1.0, sigma=sigma)

        # Dead chunk check (just a safety threshold to prevent SINDy scaling blowout)
        if np.std(X_np[:, 1]) < 1e-8:
            return np.nan, np.nan, np.nan

        # SINDy model
        library = ps.PolynomialLibrary(degree=degree)
        optimizer = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(
            feature_names=["theta", "omega"],
            feature_library=library,
            optimizer=optimizer,
        )
        model.fit(X_np, t=1)

        # Forward simulate
        try:
            sim = model.simulate(X_np[0], t_np, integrator="odeint")
            if np.any(np.isnan(sim)) or np.any(np.isinf(sim)):
                raise ValueError("Simulation diverged")
            if np.max(np.abs(sim[:, 1])) > 100 * (np.max(np.abs(X_np[:, 1])) + 1e-10):
                raise ValueError("Simulation blew up")
            sim_rmse_omega = float(np.sqrt(np.mean((sim[:, 1] - X_np[:, 1])**2)))
            sim_rmse_theta = float(np.sqrt(np.mean((sim[:, 0] - X_np[:, 0])**2)))
            sim_rmse_total = float(np.sqrt((sim_rmse_omega**2 + sim_rmse_theta**2) / 2))
            return sim_rmse_omega, sim_rmse_theta, sim_rmse_total
        except Exception:
            return np.nan, np.nan, np.nan

    except Exception as e:
        print(f"    Chunk fitting failed: {e}")
        return np.nan, np.nan, np.nan


# =============================================================================
# Evaluate one hyperparameter combination across all chunks
# =============================================================================

def evaluate_hyperparams(chunks, hyperparams, verbose=True):
    """Evaluate one HP combo on all chunks. Returns (mean_rmse_omega, n_success, all_rmses)."""
    rmse_omega_list = []
    rmse_theta_list = []
    rmse_total_list = []
    n_total = len(chunks)

    for i, chunk in enumerate(chunks):
        rmse_o, rmse_t, rmse_tot = fit_single_chunk(chunk, hyperparams)
        rmse_omega_list.append(rmse_o)
        rmse_theta_list.append(rmse_t)
        rmse_total_list.append(rmse_tot)

        if verbose and (i + 1) % 50 == 0:
            valid_so_far = [r for r in rmse_omega_list if not np.isnan(r)]
            running_mean = np.mean(valid_so_far) if valid_so_far else float('nan')
            print(f"    Chunk {i+1}/{n_total} | Running mean RMSE omega: {running_mean:.6f} | Valid: {len(valid_so_far)}/{i+1}")

    valid_omega = [r for r in rmse_omega_list if not np.isnan(r)]
    valid_theta = [r for r in rmse_theta_list if not np.isnan(r)]
    valid_total = [r for r in rmse_total_list if not np.isnan(r)]

    mean_rmse_omega = np.mean(valid_omega) if valid_omega else float('nan')
    median_rmse_omega = np.median(valid_omega) if valid_omega else float('nan')
    std_rmse_omega = np.std(valid_omega) if valid_omega else float('nan')
    mean_rmse_theta = np.mean(valid_theta) if valid_theta else float('nan')
    mean_rmse_total = np.mean(valid_total) if valid_total else float('nan')
    n_success = len(valid_omega)

    return {
        "mean_rmse_omega": mean_rmse_omega,
        "median_rmse_omega": median_rmse_omega,
        "std_rmse_omega": std_rmse_omega,
        "mean_rmse_theta": mean_rmse_theta,
        "mean_rmse_total": mean_rmse_total,
        "n_success": n_success,
        "n_total": n_total,
    }


# =============================================================================
# Combo generation
# =============================================================================

def generate_all_combos(space):
    """Generate all hyperparameter combinations (full grid)."""
    keys = list(space.keys())
    values = list(space.values())
    combos = []
    for combo_values in itertools.product(*values):
        combos.append(dict(zip(keys, combo_values)))
    return combos


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SINDy hyperparameter tuning on synthetic datasets")
    parser.add_argument("--n-chunks", type=int, default=None,
                        help="Maximum number of chunks to evaluate (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk sampling (default: 42)")
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Specific combo index to evaluate (for SLURM arrays)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for the run folder")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick sanity check: 2 chunks, 3 combos")
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../SVISE/synthetic_dataset_validation/synthetic_with_wiener.npz")
    
    if not os.path.exists(data_path):
        print(f"Error: Synthetic npz file not found at {data_path}")
        return

    all_chunks = load_synthetic_data(data_path)
    
    # Apply chunk limiting or dry run logic
    if args.dry_run:
        args.n_chunks = 2
        print("=" * 60)
        print("DRY RUN MODE: n_chunks=2")
        print("=" * 60)

    if args.n_chunks is not None and args.n_chunks < len(all_chunks):
        rng = random.Random(args.seed)
        chunks = rng.sample(all_chunks, args.n_chunks)
        print(f"Randomly sampled {args.n_chunks} chunks using seed {args.seed}.")
    else:
        chunks = all_chunks

    # Generate all combos (exhaustive — 60 total)
    all_combos = generate_all_combos(HYPERPARAMETER_SPACE)
    
    if args.dry_run and len(all_combos) > 3:
        all_combos = all_combos[:3]
        
    n_combos = len(all_combos)
    n_chunks_eval = len(chunks)

    print(f"\n{'=' * 60}")
    print(f"SINDy HYPERPARAMETER TUNING — EXHAUSTIVE GRID SEARCH (SYNTHETIC, 1-HOUR CHUNKS)")
    print(f"{'=' * 60}")
    print(f"Search space: {n_combos} combinations")
    print(f"  sigma:     {HYPERPARAMETER_SPACE['sigma']}")
    print(f"  degree:    {HYPERPARAMETER_SPACE['degree']}")
    print(f"  threshold: {HYPERPARAMETER_SPACE['threshold']}")
    print(f"Chunks per combination: {n_chunks_eval}")
    if args.combo_index is not None:
        print(f"Evaluating combo index: {args.combo_index}")
    print(f"{'=' * 60}\n")

    # Output directory
    if args.run_name:
        results_dir = os.path.join(os.path.dirname(__file__), "results_sindy_synthetic_hp_tuning_1h", args.run_name)
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "results_sindy_synthetic_hp_tuning_1h", f"run_{timestamp}")

    os.makedirs(results_dir, exist_ok=True)

    # CSV file
    if args.combo_index is not None:
        csv_path = os.path.join(results_dir, f"hp_tuning_combo_{args.combo_index:03d}.csv")
    else:
        csv_path = os.path.join(results_dir, f"hp_tuning_{timestamp}.csv")

    hp_keys = list(HYPERPARAMETER_SPACE.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Combo_Index"] + [k.title() for k in hp_keys] +
            ["Mean_RMSE_Omega", "Median_RMSE_Omega", "Std_RMSE_Omega",
             "Mean_RMSE_Theta", "Mean_RMSE_Total",
             "Num_Success", "Num_Total"]
        )

    best_mean_rmse = float('inf')
    best_combo = None
    best_combo_idx = -1

    for combo_idx, hyperparams in enumerate(all_combos):
        if args.combo_index is not None and combo_idx != args.combo_index:
            continue

        print(f"\n{'=' * 60}")
        print(f"Combo {combo_idx}/{n_combos}: sigma={hyperparams['sigma']}, "
              f"degree={hyperparams['degree']}, threshold={hyperparams['threshold']:.1e}")
        print(f"{'=' * 60}")

        results = evaluate_hyperparams(chunks, hyperparams)

        print(f"  => Mean RMSE omega: {results['mean_rmse_omega']:.6f} +/- {results['std_rmse_omega']:.6f}")
        print(f"     Median RMSE omega: {results['median_rmse_omega']:.6f}")
        print(f"     Success: {results['n_success']}/{results['n_total']}")

        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [combo_idx]
            for k in hp_keys:
                v = hyperparams[k]
                if isinstance(v, float):
                    row.append(f"{v:.1e}")
                else:
                    row.append(v)
            row += [
                f"{results['mean_rmse_omega']:.6f}" if not np.isnan(results['mean_rmse_omega']) else "nan",
                f"{results['median_rmse_omega']:.6f}" if not np.isnan(results['median_rmse_omega']) else "nan",
                f"{results['std_rmse_omega']:.6f}" if not np.isnan(results['std_rmse_omega']) else "nan",
                f"{results['mean_rmse_theta']:.6f}" if not np.isnan(results['mean_rmse_theta']) else "nan",
                f"{results['mean_rmse_total']:.6f}" if not np.isnan(results['mean_rmse_total']) else "nan",
                results['n_success'],
                results['n_total'],
            ]
            writer.writerow(row)

        # Track best
        if not np.isnan(results['mean_rmse_omega']) and results['mean_rmse_omega'] < best_mean_rmse:
            best_mean_rmse = results['mean_rmse_omega']
            best_combo = hyperparams.copy()
            best_combo_idx = combo_idx

    # Save combo-level JSON summary
    elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60

    if args.combo_index is not None and best_combo is not None:
        combo_json_path = os.path.join(results_dir, f"combo_{args.combo_index:03d}.json")
        with open(combo_json_path, 'w') as f:
            json.dump({
                "combo_index": args.combo_index,
                "hyperparams": best_combo,
                "mean_rmse_omega": best_mean_rmse,
                "n_chunks": n_chunks_eval,
                "elapsed_minutes": elapsed,
                "timestamp": timestamp,
            }, f, indent=4)
        print(f"\nCombo JSON saved to: {combo_json_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"SINDy HP TUNING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {csv_path}")
    print(f"Elapsed: {elapsed:.1f} minutes")

    if best_combo is not None:
        print(f"\nBest combination (combo #{best_combo_idx}):")
        for k, v in best_combo.items():
            print(f"  {k}: {v}")
        print(f"  Mean RMSE omega: {best_mean_rmse:.6f}")

        if args.combo_index is None:
            best_json_path = os.path.join(results_dir, f"best_hyperparams_{timestamp}.json")
            with open(best_json_path, 'w') as f:
                json.dump({
                    "best_hyperparams": best_combo,
                    "mean_rmse_omega": best_mean_rmse,
                    "combo_index": best_combo_idx,
                    "n_combos": n_combos,
                    "n_chunks": n_chunks_eval,
                    "timestamp": timestamp,
                    "elapsed_minutes": elapsed,
                }, f, indent=4)
            print(f"Best hyperparams saved to: {best_json_path}")
    else:
        print("\nNo valid results found.")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
