"""
SVISE Final Evaluation: All Valid 5-Minute Chunks

Trains the SVISE model with the best hyperparameters (from tuning) on a range
of valid 5-min chunks. Designed to run as part of a SLURM array job.

Usage:
    # Process chunks 0-999:
    python run_analysis_5min_all_chunks.py --start-chunk 0 --end-chunk 999

    # Process all chunks (single node, very slow):
    python run_analysis_5min_all_chunks.py
"""
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

# Ensure svise is in path if not installed as package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

# =============================================================================
# BEST HYPERPARAMETERS (from tuning run 20260303_020514)
# =============================================================================
BEST_HYPERPARAMS = {
    "model": "integrator",
    "sigma": 10,
    "degree": 3,
    "tau": 0.5,
    "lr": 0.001,
    "n_tau": 100,
    "measurement_noise": 1e-5,
    "n_reparam_samples": 30,
}

# Early stopping config (same as tuning)
MAX_EPOCHS = 10000
PATIENCE = 300

# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path}...")
    data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)

    return data


def get_all_valid_chunks(data):
    """Get ALL valid 5-minute chunks from the entire dataset.

    A valid chunk has exactly 300 samples (1 Hz for 5 minutes) with no missing data.
    Returns a list of (chunk_start_time, chunk_dataframe) tuples.
    """
    print("Finding all valid 5-minute chunks in the dataset...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()

    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == 300:
            valid_chunks.append((chunk_start, group))

    if not valid_chunks:
        raise ValueError("No valid 5-minute chunks found in the dataset.")

    print(f"Found {len(valid_chunks)} valid 5-minute chunks in total.")
    return valid_chunks


def prepare_data(chunk_df, dt=1.0, sigma=10):
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

def train_single_chunk(chunk_df, hp=BEST_HYPERPARAMS):
    """Train one SVISE model on one chunk.

    Returns dict with: rmse_omega, rmse_theta, rmse_total, final_loss, equations, stopped_epoch
    """
    DT = 1.0

    try:
        t_np, X_np, _ = prepare_data(chunk_df, dt=DT, sigma=hp["sigma"])

        # Convert to Torch Tensors
        train_t = torch.tensor(t_np, dtype=torch.float32)
        train_x = torch.tensor(X_np, dtype=torch.float32)

        # Global Scaling
        mean_x = train_x.mean(dim=0)
        std_x = train_x.std(dim=0)
        std_x[std_x < 1e-6] = 1.0

        t_scale = 30.0

        if hp["model"] == "integrator":
            mean_x[1] = 0.0
            std_x[0] = std_x[1] * t_scale

        train_x_scaled = (train_x - mean_x) / std_x

        # Model Setup
        d = 2
        num_meas = 2
        G = torch.eye(d)
        measurement_noise = torch.tensor([hp["measurement_noise"], hp["measurement_noise"]])

        train_t_scaled = train_t / t_scale
        t_span = (train_t_scaled[0], train_t_scaled[-1])

        common_params = {
            "d": d,
            "t_span": t_span,
            "degree": hp["degree"],
            "n_reparam_samples": hp["n_reparam_samples"],
            "G": G,
            "num_meas": num_meas,
            "measurement_noise": measurement_noise,
            "tau": hp["tau"],
            "train_t": train_t_scaled,
            "train_x": train_x_scaled,
            "input_labels": ["theta", "omega"],
        }

        if hp["model"] == "sparse":
            model = SparsePolynomialSDE(**common_params, n_tau=hp["n_tau"])
        elif hp["model"] == "integrator":
            model = SparsePolynomialIntegratorSDE(**common_params, n_tau=hp["n_tau"])
        else:
            raise ValueError(f"Unknown model type: {hp['model']}")

        # Training with early stopping
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
        best_loss = float('inf')
        epochs_without_improvement = 0
        nan_recoveries = 0
        max_nan_recoveries = 5
        current_lr = hp["lr"]
        stopped_epoch = 0

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
                    stopped_epoch = epoch
                    break
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_loss = loss.item()

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

            if epochs_without_improvement >= PATIENCE:
                stopped_epoch = epoch
                break

            stopped_epoch = epoch

        # Restore best checkpoint
        model.load_state_dict(copy.deepcopy(best_checkpoint['model']))
        model.sde_prior.resample_weights()

        # Extract equations
        equations = []
        try:
            equation_strings = model.sde_prior.get_feature_names()
            equations = equation_strings
        except Exception as e:
            equations = [f"Error: {e}"] * d

        # Calculate RMSE
        with torch.no_grad():
            x_pred_scaled = model.marginal_sde.mean(train_t_scaled)
            x_pred = x_pred_scaled * std_x + mean_x

            mse_theta = ((x_pred[:, 0] - train_x[:, 0]) ** 2).mean()
            mse_omega = ((x_pred[:, 1] - train_x[:, 1]) ** 2).mean()

            rmse_theta = torch.sqrt(mse_theta).item()
            rmse_omega = torch.sqrt(mse_omega).item()
            rmse_total = torch.sqrt((mse_theta + mse_omega) / 2).item()

        return {
            "rmse_omega": rmse_omega,
            "rmse_theta": rmse_theta,
            "rmse_total": rmse_total,
            "final_loss": best_checkpoint['loss'],
            "stopped_epoch": stopped_epoch,
            "nan_recoveries": nan_recoveries,
            "equations": equations,
            "scaling_params": {
                "mean_x": mean_x.tolist(),
                "std_x": std_x.tolist(),
                "t_scale": t_scale,
            },
        }

    except Exception as e:
        print(f"    Chunk training failed: {e}")
        return {
            "rmse_omega": float('nan'),
            "rmse_theta": float('nan'),
            "rmse_total": float('nan'),
            "final_loss": float('nan'),
            "stopped_epoch": -1,
            "nan_recoveries": -1,
            "equations": [f"FAILED: {e}"],
            "scaling_params": None,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SVISE final evaluation on all valid 5-min chunks")
    parser.add_argument("--start-chunk", type=int, default=0,
                        help="Start chunk index (inclusive, 0-indexed)")
    parser.add_argument("--end-chunk", type=int, default=-1,
                        help="End chunk index (inclusive, 0-indexed). -1 = last chunk.")
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load data
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    data = load_data(DATA_PATH)
    all_chunks = get_all_valid_chunks(data)
    total_chunks = len(all_chunks)

    # Determine chunk range
    start = args.start_chunk
    end = args.end_chunk if args.end_chunk >= 0 else total_chunks - 1
    end = min(end, total_chunks - 1)

    if start > end or start >= total_chunks:
        print(f"Error: Invalid chunk range [{start}, {end}]. Total chunks: {total_chunks}")
        return

    n_chunks = end - start + 1

    print(f"\n{'=' * 60}")
    print(f"SVISE FINAL EVALUATION — ALL CHUNKS")
    print(f"{'=' * 60}")
    print(f"Hyperparameters: {json.dumps(BEST_HYPERPARAMS, indent=2)}")
    print(f"Total valid chunks in dataset: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"Early stopping: patience={PATIENCE}, max_epochs={MAX_EPOCHS}")
    print(f"{'=' * 60}\n")

    # Output directory
    results_dir = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks")
    os.makedirs(results_dir, exist_ok=True)

    # Per-chunk CSV (append-safe for resumability)
    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Chunk_Index", "Chunk_Start_Time",
            "RMSE_Omega", "RMSE_Theta", "RMSE_Total",
            "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
            "Eq_Theta", "Eq_Omega"
        ])

    # Process chunks
    rmse_omega_list = []
    rmse_theta_list = []
    rmse_total_list = []
    loss_list = []

    for i in range(start, end + 1):
        chunk_start_time, chunk_df = all_chunks[i]
        progress = i - start + 1

        print(f"\n--- Chunk {i} ({progress}/{n_chunks}) | Start: {chunk_start_time} ---")

        result = train_single_chunk(chunk_df)

        # Collect stats
        rmse_omega_list.append(result["rmse_omega"])
        rmse_theta_list.append(result["rmse_theta"])
        rmse_total_list.append(result["rmse_total"])
        loss_list.append(result["final_loss"])

        # Extract equations
        eq_theta = result["equations"][0] if len(result["equations"]) > 0 else "N/A"
        eq_omega = result["equations"][1] if len(result["equations"]) > 1 else "N/A"

        print(f"    RMSE omega: {result['rmse_omega']:.6f} | "
              f"RMSE theta: {result['rmse_theta']:.6f} | "
              f"Loss: {result['final_loss']:.4f} | "
              f"Epoch: {result['stopped_epoch']}")
        print(f"    Eq theta: {eq_theta}")
        print(f"    Eq omega: {eq_omega}")

        # Append to CSV immediately (crash-safe)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, str(chunk_start_time),
                f"{result['rmse_omega']:.6f}" if not np.isnan(result['rmse_omega']) else "nan",
                f"{result['rmse_theta']:.6f}" if not np.isnan(result['rmse_theta']) else "nan",
                f"{result['rmse_total']:.6f}" if not np.isnan(result['rmse_total']) else "nan",
                f"{result['final_loss']:.4f}" if not np.isnan(result['final_loss']) else "nan",
                result['stopped_epoch'],
                result['nan_recoveries'],
                eq_theta,
                eq_omega,
            ])

        # Print running stats every 50 chunks
        if progress % 50 == 0 or progress == n_chunks:
            valid_omegas = [r for r in rmse_omega_list if not np.isnan(r)]
            valid_losses = [r for r in loss_list if not np.isnan(r)]
            mean_rmse = np.mean(valid_omegas) if valid_omegas else float('nan')
            mean_loss = np.mean(valid_losses) if valid_losses else float('nan')
            elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60
            rate = elapsed / progress
            eta = rate * (n_chunks - progress)
            print(f"\n  [Progress {progress}/{n_chunks}] "
                  f"Mean RMSE omega: {mean_rmse:.6f} | "
                  f"Mean loss: {mean_loss:.4f} | "
                  f"Valid: {len(valid_omegas)}/{progress} | "
                  f"Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

    # Final summary for this range
    valid_omegas = [r for r in rmse_omega_list if not np.isnan(r)]
    valid_thetas = [r for r in rmse_theta_list if not np.isnan(r)]
    valid_totals = [r for r in rmse_total_list if not np.isnan(r)]
    valid_losses = [r for r in loss_list if not np.isnan(r)]

    summary = {
        "hyperparams": BEST_HYPERPARAMS,
        "chunk_range": [start, end],
        "n_chunks_processed": n_chunks,
        "n_chunks_succeeded": len(valid_omegas),
        "n_chunks_failed": n_chunks - len(valid_omegas),
        "mean_rmse_omega": float(np.mean(valid_omegas)) if valid_omegas else float('nan'),
        "std_rmse_omega": float(np.std(valid_omegas)) if valid_omegas else float('nan'),
        "mean_rmse_theta": float(np.mean(valid_thetas)) if valid_thetas else float('nan'),
        "std_rmse_theta": float(np.std(valid_thetas)) if valid_thetas else float('nan'),
        "mean_rmse_total": float(np.mean(valid_totals)) if valid_totals else float('nan'),
        "mean_loss": float(np.mean(valid_losses)) if valid_losses else float('nan'),
        "std_loss": float(np.std(valid_losses)) if valid_losses else float('nan'),
        "timestamp": timestamp,
        "csv_file": csv_filename,
        "early_stopping": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE},
        "elapsed_minutes": (datetime.datetime.now() - start_time).total_seconds() / 60,
    }

    summary_path = os.path.join(results_dir, f"summary_{start:05d}_to_{end:05d}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE — Chunks [{start}, {end}]")
    print(f"{'=' * 60}")
    print(f"Chunks processed: {n_chunks}")
    print(f"Chunks succeeded: {len(valid_omegas)} ({100*len(valid_omegas)/n_chunks:.1f}%)")
    print(f"Mean RMSE omega:  {summary['mean_rmse_omega']:.6f} ± {summary['std_rmse_omega']:.6f}")
    print(f"Mean RMSE theta:  {summary['mean_rmse_theta']:.6f} ± {summary['std_rmse_theta']:.6f}")
    print(f"Mean RMSE total:  {summary['mean_rmse_total']:.6f}")
    print(f"Mean Loss:        {summary['mean_loss']:.4f} ± {summary['std_loss']:.4f}")
    print(f"Elapsed time:     {summary['elapsed_minutes']:.1f} minutes")
    print(f"\nPer-chunk CSV: {csv_path}")
    print(f"Summary JSON:  {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
