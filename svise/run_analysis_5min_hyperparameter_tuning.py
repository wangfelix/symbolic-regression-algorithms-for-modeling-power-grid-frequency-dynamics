import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
import argparse
import csv
import json
import datetime
import copy
import itertools
import random

# Ensure svise is in path if not installed as package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

# =============================================================================
# HYPERPARAMETER SEARCH SPACE — Edit these values to control the search space
# =============================================================================
HYPERPARAMETER_SPACE = {
    "model": ["integrator"],
    "sigma": [0, 5, 10],
    "degree": [2, 3],
    "tau": [5e-1, 1e-1, 1e-2, 1e-3, 1e-5],
    "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "n_tau": [50, 100, 200],
    "measurement_noise": [1e-3, 1e-4, 1e-5],
    "n_reparam_samples": [15, 30, 50],
}

# Number of random combinations to sample (set to None to do full grid search)
N_RANDOM_SAMPLES = 150

# Early stopping config
MAX_EPOCHS = 10000
PATIENCE = 300  # Stop if loss hasn't improved for this many epochs


# =============================================================================
# Data Loading (reused from run_analysis_5min_chunks.py)
# =============================================================================

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path}...")
    data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:,'freq'] = data.loc[:,'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(),'QI'] = 2
        data.loc[~data['freq'].isna(),'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)

    return data


def get_valid_chunks_9_to_10(data, max_chunks=5000):
    """Get valid 5-minute chunks whose start time falls in the 9:00-10:00 window.

    A valid chunk has exactly 300 samples (1 Hz for 5 minutes) with no missing data.
    Chunks starting at 9:00, 9:05, 9:10, ..., 9:55 are included.
    """
    print("Filtering for valid 5-minute chunks in the 9:00-10:00 window...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()

    # Group by 5min intervals
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        # Must be a full 5-min chunk (300 samples at 1Hz)
        if len(group) != 300:
            continue
        # Filter for 9:00-10:00 window (hour == 9)
        if chunk_start.hour != 9:
            continue
        valid_chunks.append(group)
        if len(valid_chunks) >= max_chunks:
            break

    if not valid_chunks:
        raise ValueError("No valid 5-minute chunks found in the 9:00-10:00 window.")

    print(f"Found {len(valid_chunks)} valid 9:00-10:00 chunks (max requested: {max_chunks}).")
    return valid_chunks


def prepare_data(chunk_df, dt=1.0, sigma=5):
    freq_values = chunk_df['freq'].values

    # Calculate Frequency Deviation
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    # Gaussian Filter
    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega = omega_raw

    # Theta (Integral of omega)
    theta = np.cumsum(omega) * dt

    # Time vector
    t = np.arange(len(omega)) * dt

    # Combine into State Matrix X = [theta, omega]
    X = np.stack([theta, omega], axis=1)

    return t, X, omega_raw


# =============================================================================
# Training a single chunk
# =============================================================================

def train_single_chunk(chunk_df, hyperparams):
    """Train one SVISE model on one chunk. Returns rmse_omega or NaN on failure."""
    DT = 1.0
    sigma = hyperparams["sigma"]
    model_type = hyperparams["model"]
    degree = hyperparams["degree"]
    tau = hyperparams["tau"]
    lr = hyperparams["lr"]
    n_tau = hyperparams["n_tau"]
    meas_noise_val = hyperparams["measurement_noise"]
    n_reparam = hyperparams["n_reparam_samples"]

    try:
        t_np, X_np, _ = prepare_data(chunk_df, dt=DT, sigma=sigma)

        # Convert to Torch Tensors
        train_t = torch.tensor(t_np, dtype=torch.float32)
        train_x = torch.tensor(X_np, dtype=torch.float32)

        # Global Scaling
        mean_x = train_x.mean(dim=0)
        std_x = train_x.std(dim=0)
        std_x[std_x < 1e-6] = 1.0

        t_scale = 30.0

        if model_type == "integrator":
            mean_x[1] = 0.0
            std_x[0] = std_x[1] * t_scale

        train_x_scaled = (train_x - mean_x) / std_x

        # Model Setup
        d = 2
        num_meas = 2
        G = torch.eye(d)
        measurement_noise = torch.tensor([meas_noise_val, meas_noise_val])

        train_t_scaled = train_t / t_scale
        t_span = (train_t_scaled[0], train_t_scaled[-1])

        common_params = {
            "d": d,
            "t_span": t_span,
            "degree": degree,
            "n_reparam_samples": n_reparam,
            "G": G,
            "num_meas": num_meas,
            "measurement_noise": measurement_noise,
            "tau": tau,
            "train_t": train_t_scaled,
            "train_x": train_x_scaled,
            "input_labels": ["theta", "omega"],
        }

        if model_type == "sparse":
            model = SparsePolynomialSDE(**common_params, n_tau=n_tau)
        elif model_type == "integrator":
            model = SparsePolynomialIntegratorSDE(**common_params, n_tau=n_tau)
        else:
            return float('nan')

        # Training with early stopping
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss = float('inf')
        epochs_without_improvement = 0
        nan_recoveries = 0
        max_nan_recoveries = 5
        current_lr = lr

        best_checkpoint = {
            'model': copy.deepcopy(model.state_dict()),
            'epoch': -1,
            'loss': float('inf'),
        }

        for epoch in range(MAX_EPOCHS):
            optimizer.zero_grad()

            loss = -model.elbo(train_t_scaled, train_x_scaled, beta=1.0, N=len(train_t))

            if torch.isnan(loss) or torch.isinf(loss):
                nan_recoveries += 1
                current_lr *= 0.5
                model.load_state_dict(copy.deepcopy(best_checkpoint['model']))
                model.sde_prior.resample_weights()
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

                if nan_recoveries >= max_nan_recoveries:
                    break
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()

            # Track best loss for early stopping
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                epochs_without_improvement = 0
                best_checkpoint = {
                    'model': copy.deepcopy(model.state_dict()),
                    'epoch': epoch,
                    'loss': current_loss,
                }
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= PATIENCE:
                break

        # Restore best checkpoint
        model.load_state_dict(copy.deepcopy(best_checkpoint['model']))
        model.sde_prior.resample_weights()

        # Calculate RMSE (omega only)
        with torch.no_grad():
            x_pred_scaled = model.marginal_sde.mean(train_t_scaled)
            x_pred = x_pred_scaled * std_x + mean_x
            mse_omega = ((x_pred[:, 1] - train_x[:, 1]) ** 2).mean()
            rmse_omega = torch.sqrt(mse_omega).item()

        return rmse_omega

    except Exception as e:
        print(f"    Chunk training failed: {e}")
        return float('nan')


# =============================================================================
# Evaluate one hyperparameter combination across all chunks
# =============================================================================

def evaluate_hyperparams(chunks, hyperparams, verbose=True):
    """Train on all chunks with the given hyperparams. Returns (mean_rmse, n_success, all_rmses)."""
    rmse_values = []
    n_total = len(chunks)

    for i, chunk_df in enumerate(chunks):
        rmse = train_single_chunk(chunk_df, hyperparams)
        rmse_values.append(rmse)

        if verbose and (i + 1) % max(1, n_total // 10) == 0:
            valid = [r for r in rmse_values if not np.isnan(r)]
            running_mean = np.mean(valid) if valid else float('nan')
            print(f"    Chunk {i+1}/{n_total}: running mean RMSE = {running_mean:.6f} ({len(valid)} valid)")

    valid_rmses = [r for r in rmse_values if not np.isnan(r)]
    mean_rmse = np.mean(valid_rmses) if valid_rmses else float('nan')
    n_success = len(valid_rmses)

    return mean_rmse, n_success, rmse_values


# =============================================================================
# Combo generation: full grid or randomized sampling
# =============================================================================

def generate_all_combos(space):
    """Generate all hyperparameter combinations (full grid)."""
    keys = list(space.keys())
    values = list(space.values())
    for combo_values in itertools.product(*values):
        yield dict(zip(keys, combo_values))


def sample_random_combos(space, n_samples, seed=42):
    """Randomly sample n_samples unique combinations from the search space."""
    rng = random.Random(seed)
    keys = list(space.keys())
    values = list(space.values())

    # Total possible combos
    total_possible = 1
    for v in values:
        total_possible *= len(v)

    if n_samples >= total_possible:
        print(f"Requested {n_samples} samples but only {total_possible} unique combos exist. Using full grid.")
        return list(generate_all_combos(space))

    # Sample unique combos using index-based sampling to guarantee uniqueness
    sampled_indices = rng.sample(range(total_possible), n_samples)
    combos = []
    for idx in sampled_indices:
        combo = {}
        remaining = idx
        for key, vals in zip(keys, values):
            combo[key] = vals[remaining % len(vals)]
            remaining //= len(vals)
        combos.append(combo)

    return combos


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for SVISE 5-min chunk analysis")
    parser.add_argument("--max-chunks", type=int, default=5000,
                        help="Maximum number of 9:00-10:00 chunks to use (default: 5000)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of random combos to sample (default: uses N_RANDOM_SAMPLES from script, set to 0 for full grid)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick sanity check: use only 2 chunks, 5 epochs, and 3 combos")
    args = parser.parse_args()

    # Dry-run overrides
    global MAX_EPOCHS, PATIENCE
    if args.dry_run:
        MAX_EPOCHS = 5
        PATIENCE = 3
        args.max_chunks = min(args.max_chunks, 2)
        print("=" * 60)
        print("DRY RUN MODE: max_epochs=5, patience=3, max_chunks=2")
        print("=" * 60)

    # Determine number of samples
    n_samples = args.n_samples if args.n_samples is not None else N_RANDOM_SAMPLES
    if args.dry_run and (n_samples is None or n_samples > 3):
        n_samples = 3

    # Load data once
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    data = load_data(DATA_PATH)
    chunks = get_valid_chunks_9_to_10(data, max_chunks=args.max_chunks)

    # Generate combos
    total_grid_size = 1
    for v in HYPERPARAMETER_SPACE.values():
        total_grid_size *= len(v)

    if n_samples is not None and n_samples > 0:
        search_mode = "RANDOMIZED"
        all_combos = sample_random_combos(HYPERPARAMETER_SPACE, n_samples, seed=args.seed)
    else:
        search_mode = "FULL GRID"
        all_combos = list(generate_all_combos(HYPERPARAMETER_SPACE))

    n_combos = len(all_combos)
    n_chunks = len(chunks)

    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER {search_mode} SEARCH")
    print(f"{'=' * 60}")
    print(f"Total search space: {total_grid_size} combinations")
    print(f"Evaluating: {n_combos} combinations ({search_mode})")
    print(f"Chunks per combination: {n_chunks}")
    print(f"Total training runs: {n_combos * n_chunks}")
    print(f"Early stopping: patience={PATIENCE}, max_epochs={MAX_EPOCHS}")
    if search_mode == "RANDOMIZED":
        print(f"Random seed: {args.seed}")
    print(f"{'=' * 60}\n")

    # Results CSV
    results_dir = os.path.join(os.path.dirname(__file__), "results_hp_tuning")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"hp_tuning_{timestamp}.csv")

    # Write CSV header
    hp_keys = list(HYPERPARAMETER_SPACE.keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Combo_Index"] + [k.title() for k in hp_keys] +
            ["Mean_RMSE_Omega", "Num_Success", "Num_Total"]
        )

    best_rmse = float('inf')
    best_combo = None
    best_combo_idx = -1

    for combo_idx, hyperparams in enumerate(all_combos):
        print(f"\n{'=' * 60}")
        print(f"Combo {combo_idx + 1}/{n_combos}: {hyperparams}")
        print(f"{'=' * 60}")

        mean_rmse, n_success, _ = evaluate_hyperparams(chunks, hyperparams)

        print(f"  => Mean RMSE (omega): {mean_rmse:.6f} ({n_success}/{n_chunks} chunks succeeded)")

        # Append to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [combo_idx + 1]
            for k in hp_keys:
                v = hyperparams[k]
                if isinstance(v, float):
                    row.append(f"{v:.1e}")
                else:
                    row.append(v)
            row += [
                f"{mean_rmse:.6f}" if not np.isnan(mean_rmse) else "nan",
                n_success,
                n_chunks,
            ]
            writer.writerow(row)

        # Track best
        if not np.isnan(mean_rmse) and mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_combo = hyperparams.copy()
            best_combo_idx = combo_idx + 1

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"{search_mode} SEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {csv_path}")

    if best_combo is not None:
        print(f"\nBest combination (combo #{best_combo_idx}):")
        for k, v in best_combo.items():
            print(f"  {k}: {v}")
        print(f"  Mean RMSE (omega): {best_rmse:.6f}")

        # Save best result as JSON
        best_json_path = os.path.join(results_dir, f"best_hyperparams_{timestamp}.json")
        with open(best_json_path, 'w') as f:
            json.dump({
                "best_hyperparams": {k: str(v) if isinstance(v, float) else v for k, v in best_combo.items()},
                "best_hyperparams_raw": best_combo,
                "mean_rmse_omega": best_rmse,
                "combo_index": best_combo_idx,
                "n_combos_evaluated": n_combos,
                "total_search_space": total_grid_size,
                "search_mode": search_mode.lower(),
                "n_chunks": n_chunks,
                "timestamp": timestamp,
                "seed": args.seed,
                "early_stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE},
            }, f, indent=4)
        print(f"Best hyperparams saved to: {best_json_path}")
    else:
        print("\nNo valid results found. All combinations failed.")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
