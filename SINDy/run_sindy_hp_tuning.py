"""
SINDy Hyperparameter Tuning on 9:00-10:00 5-Minute Chunks

Exhaustive grid search over sigma, degree, and STLSQ threshold.
Each SLURM array task evaluates one hyperparameter combination
across all valid 9:00-10:00 chunks.

Usage:
    python run_sindy_hp_tuning.py --combo-index 0
    python run_sindy_hp_tuning.py --combo-index 0 --run-name run_SLURM_12345
    python run_sindy_hp_tuning.py  # evaluate all combos sequentially
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
# Data Loading (reused from run_sindy_5min_all_chunks.py)
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


def get_valid_chunks_9_to_10(data, max_chunks=5000, seed=42):
    """Get valid 5-minute chunks whose start time falls in the 9:00-10:00 window."""
    print("Filtering for valid 5-minute chunks in the 9:00-10:00 window...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) != 300:
            continue
        if chunk_start.hour != 9:
            continue
        valid_chunks.append(group)

    if not valid_chunks:
        raise ValueError("No valid 5-minute chunks found in the 9:00-10:00 window.")

    if len(valid_chunks) > max_chunks:
        rng = random.Random(seed)
        sampled_chunks = rng.sample(valid_chunks, max_chunks)
        print(f"Found {len(valid_chunks)} total valid 9:00-10:00 chunks.")
        print(f"Randomly sampled {max_chunks} chunks using seed {seed}.")
        return sampled_chunks
    else:
        print(f"Found {len(valid_chunks)} valid 9:00-10:00 chunks (max requested: {max_chunks}).")
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

def fit_single_chunk(chunk_df, hyperparams):
    """Fit SINDy on one 5-min chunk with given hyperparameters.

    Returns (sim_rmse_omega, sim_rmse_theta, sim_rmse_total) or NaNs on failure.
    """
    sigma = hyperparams["sigma"]
    degree = hyperparams["degree"]
    threshold = hyperparams["threshold"]

    try:
        t_np, X_np, omega_raw = prepare_data(chunk_df, dt=1.0, sigma=sigma)

        # Dead chunk check
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

    for i, chunk_df in enumerate(chunks):
        rmse_o, rmse_t, rmse_tot = fit_single_chunk(chunk_df, hyperparams)
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
    parser = argparse.ArgumentParser(description="SINDy hyperparameter tuning on 9:00-10:00 chunks")
    parser.add_argument("--max-chunks", type=int, default=5000,
                        help="Maximum number of 9:00-10:00 chunks to use (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk sampling (default: 42)")
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Specific combo index to evaluate (for SLURM arrays)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for the run folder")
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
    chunks = get_valid_chunks_9_to_10(data, max_chunks=args.max_chunks, seed=args.seed)

    # Generate all combos (exhaustive — only 70 total)
    all_combos = generate_all_combos(HYPERPARAMETER_SPACE)
    n_combos = len(all_combos)
    n_chunks = len(chunks)

    print(f"\n{'=' * 60}")
    print(f"SINDy HYPERPARAMETER TUNING — EXHAUSTIVE GRID SEARCH")
    print(f"{'=' * 60}")
    print(f"Search space: {n_combos} combinations")
    print(f"  sigma:     {HYPERPARAMETER_SPACE['sigma']}")
    print(f"  degree:    {HYPERPARAMETER_SPACE['degree']}")
    print(f"  threshold: {HYPERPARAMETER_SPACE['threshold']}")
    print(f"Chunks per combination: {n_chunks}")
    if args.combo_index is not None:
        print(f"Evaluating combo index: {args.combo_index}")
    print(f"{'=' * 60}\n")

    # Output directory
    if args.run_name:
        results_dir = os.path.join(os.path.dirname(__file__), "results_sindy_hp_tuning", args.run_name)
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "results_sindy_hp_tuning", f"run_{timestamp}")

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
                "n_chunks": n_chunks,
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
                    "n_chunks": n_chunks,
                    "timestamp": timestamp,
                    "elapsed_minutes": elapsed,
                }, f, indent=4)
            print(f"Best hyperparams saved to: {best_json_path}")
    else:
        print("\nNo valid results found.")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
