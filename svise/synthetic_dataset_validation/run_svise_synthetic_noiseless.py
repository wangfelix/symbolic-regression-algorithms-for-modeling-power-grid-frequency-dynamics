"""
SVISE Final Evaluation on Synthetic Noiseless Dataset: All Chunks.

Trains the SVISE model with the best hyperparameters (from tuning) on a range
of 5-min chunks from the noiseless synthetic dataset. Designed for SLURM array jobs.

Extra outputs compared to real-data pipeline:
  - Individual polynomial coefficients (Coeff_Intercept, Coeff_Theta, Coeff_Omega, etc.)
  - Diffusion/noise values from model.diffusion_prior.process_noise_diag

Usage:
    # Process chunks 0-499:
    python run_svise_synthetic_noiseless.py --start-chunk 0 --end-chunk 499

    # Process all chunks:
    python run_svise_synthetic_noiseless.py
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

# Ensure svise is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

# =============================================================================
# BEST HYPERPARAMETERS (copy from HP tuning results)
# =============================================================================
BEST_HYPERPARAMS = {
    "model": "integrator",
    "sigma": 0,
    "degree": 1,          # FILL after HP tuning
    "tau": 0.5,           # FILL after HP tuning
    "lr": 0.001,          # FILL after HP tuning
    "n_tau": 100,         # FILL after HP tuning
    "measurement_noise": 1e-5,   # FILL after HP tuning
    "n_reparam_samples": 30,     # FILL after HP tuning
}

# Early stopping config
MAX_EPOCHS = 10000
PATIENCE = 300


# =============================================================================
# Data Loading
# =============================================================================

def load_synthetic_data(data_path):
    """Load synthetic data and chunk into 5-minute windows."""
    print(f"Loading synthetic data from {data_path}...")
    df = pd.read_pickle(data_path)
    omega = df['omega'].values
    theta = df['theta'].values
    print(f"  Total samples: {len(omega)}")

    chunk_size = 300
    n_chunks = len(omega) // chunk_size
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append({
            'omega': omega[start:end],
            'theta': theta[start:end],
        })

    print(f"  Total 5-min chunks: {len(chunks)}")
    return chunks


def prepare_synthetic_chunk(chunk, dt=1.0, sigma=0):
    """Prepare a synthetic chunk for SVISE training."""
    omega_raw = chunk['omega'].copy()

    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega = omega_raw.copy()

    theta = np.cumsum(omega) * dt
    t = np.arange(len(omega)) * dt
    X = np.stack([theta, omega], axis=1)

    return t, X, omega_raw


# =============================================================================
# Training a single chunk
# =============================================================================

def train_single_chunk(chunk, hp=BEST_HYPERPARAMS):
    """Train one SVISE model on one synthetic chunk.

    Returns dict with: rmse_omega, rmse_theta, rmse_total, final_loss, equations,
                       stopped_epoch, coefficients, diffusion_diag
    """
    DT = 1.0

    try:
        t_np, X_np, _ = prepare_synthetic_chunk(chunk, dt=DT, sigma=hp["sigma"])

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

        # Extract polynomial coefficients (omega equation only for integrator)
        coefficients = {}
        try:
            # Get feature names and weights from the SDE prior
            sde_prior = model.sde_prior
            # For integrator model, the polynomial is for the omega equation
            if hasattr(sde_prior, 'glm') and hasattr(sde_prior.glm, 'features'):
                glm = sde_prior.glm
            elif hasattr(sde_prior, 'sparse_glm'):
                glm = sde_prior.sparse_glm
            else:
                glm = sde_prior

            # Try to get individual coefficients from the weight matrix
            if hasattr(glm, 'weight_mean'):
                weights = glm.weight_mean.detach().cpu().numpy()
            elif hasattr(glm, 'linear') and hasattr(glm.linear, 'weight'):
                weights = glm.linear.weight.detach().cpu().numpy()
            else:
                weights = None

            if weights is not None:
                # For degree=1 integrator: features are [1, theta, omega]
                # For degree=2: [1, theta, omega, theta^2, theta*omega, omega^2]
                # For degree=3: adds cubic terms
                feature_labels = _get_polynomial_feature_labels(hp["degree"])
                # weights shape depends on model; extract omega equation row
                if weights.ndim == 2:
                    # For integrator, there's typically 1 row (omega equation)
                    omega_weights = weights[0] if weights.shape[0] == 1 else weights[-1]
                else:
                    omega_weights = weights

                for i, label in enumerate(feature_labels):
                    if i < len(omega_weights):
                        coefficients[f"Coeff_{label}"] = float(omega_weights[i])
        except Exception as e:
            coefficients["coeff_error"] = str(e)

        # Extract diffusion/noise
        diffusion_diag = []
        try:
            if hasattr(model, 'diffusion_prior') and hasattr(model.diffusion_prior, 'process_noise_diag'):
                noise = model.diffusion_prior.process_noise_diag.detach().cpu().numpy()
                diffusion_diag = noise.tolist()
        except Exception as e:
            diffusion_diag = [f"Error: {e}"]

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
            "coefficients": coefficients,
            "diffusion_diag": diffusion_diag,
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
            "coefficients": {},
            "diffusion_diag": [],
            "scaling_params": None,
        }


def _get_polynomial_feature_labels(degree):
    """Get polynomial feature labels for [theta, omega] up to given degree."""
    labels = ["Intercept", "Theta", "Omega"]
    if degree >= 2:
        labels += ["Theta^2", "Theta*Omega", "Omega^2"]
    if degree >= 3:
        labels += ["Theta^3", "Theta^2*Omega", "Theta*Omega^2", "Omega^3"]
    return labels


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SVISE evaluation on all synthetic noiseless chunks")
    parser.add_argument("--start-chunk", type=int, default=0,
                        help="Start chunk index (inclusive, 0-indexed)")
    parser.add_argument("--end-chunk", type=int, default=-1,
                        help="End chunk index (inclusive, 0-indexed). -1 = last chunk.")
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Load synthetic data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "synthetic_data_noiseless.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
        print("Run generate_synthetic_data.py first.")
        return

    all_chunks = load_synthetic_data(data_path)
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
    print(f"SVISE EVALUATION — SYNTHETIC NOISELESS — ALL CHUNKS")
    print(f"{'=' * 60}")
    print(f"Hyperparameters: {json.dumps(BEST_HYPERPARAMS, indent=2)}")
    print(f"Total chunks in dataset: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"Early stopping: patience={PATIENCE}, max_epochs={MAX_EPOCHS}")
    print(f"{'=' * 60}\n")

    # Output directory
    results_dir = os.path.join(script_dir, "results_synthetic_noiseless")
    os.makedirs(results_dir, exist_ok=True)

    # Build CSV header with coefficient columns
    feature_labels = _get_polynomial_feature_labels(BEST_HYPERPARAMS["degree"])
    coeff_columns = [f"Coeff_{label}" for label in feature_labels]

    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    csv_header = [
        "Chunk_Index",
        "RMSE_Omega", "RMSE_Theta", "RMSE_Total",
        "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
        "Eq_Theta", "Eq_Omega",
    ] + coeff_columns + [
        "Diffusion_Theta", "Diffusion_Omega",
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # Process chunks
    rmse_omega_list = []
    rmse_theta_list = []
    rmse_total_list = []
    loss_list = []

    for i in range(start, end + 1):
        chunk = all_chunks[i]
        progress = i - start + 1

        print(f"\n--- Chunk {i} ({progress}/{n_chunks}) ---")

        result = train_single_chunk(chunk)

        # Collect stats
        rmse_omega_list.append(result["rmse_omega"])
        rmse_theta_list.append(result["rmse_theta"])
        rmse_total_list.append(result["rmse_total"])
        loss_list.append(result["final_loss"])

        # Extract equations
        eq_theta = result["equations"][0] if len(result["equations"]) > 0 else "N/A"
        eq_omega = result["equations"][1] if len(result["equations"]) > 1 else "N/A"

        # Extract coefficients
        coeff_values = []
        for label in feature_labels:
            key = f"Coeff_{label}"
            val = result["coefficients"].get(key, float('nan'))
            coeff_values.append(f"{val:.8e}" if not np.isnan(val) else "nan")

        # Extract diffusion
        diff_theta = result["diffusion_diag"][0] if len(result["diffusion_diag"]) > 0 and not isinstance(result["diffusion_diag"][0], str) else "nan"
        diff_omega = result["diffusion_diag"][1] if len(result["diffusion_diag"]) > 1 and not isinstance(result["diffusion_diag"][1], str) else "nan"

        print(f"    RMSE omega: {result['rmse_omega']:.6f} | "
              f"RMSE theta: {result['rmse_theta']:.6f} | "
              f"Loss: {result['final_loss']:.4f} | "
              f"Epoch: {result['stopped_epoch']}")
        print(f"    Eq theta: {eq_theta}")
        print(f"    Eq omega: {eq_omega}")
        if result["coefficients"]:
            coeff_str = ", ".join(f"{k}={v:.4e}" for k, v in result["coefficients"].items() if not k.startswith("coeff_error"))
            print(f"    Coefficients: {coeff_str}")
        if result["diffusion_diag"] and not isinstance(result["diffusion_diag"][0], str):
            print(f"    Diffusion diag: {result['diffusion_diag']}")

        # Append to CSV immediately (crash-safe)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                i,
                f"{result['rmse_omega']:.6f}" if not np.isnan(result['rmse_omega']) else "nan",
                f"{result['rmse_theta']:.6f}" if not np.isnan(result['rmse_theta']) else "nan",
                f"{result['rmse_total']:.6f}" if not np.isnan(result['rmse_total']) else "nan",
                f"{result['final_loss']:.4f}" if not np.isnan(result['final_loss']) else "nan",
                result['stopped_epoch'],
                result['nan_recoveries'],
                eq_theta,
                eq_omega,
            ] + coeff_values + [
                f"{diff_theta:.8e}" if isinstance(diff_theta, (int, float)) else diff_theta,
                f"{diff_omega:.8e}" if isinstance(diff_omega, (int, float)) else diff_omega,
            ]
            writer.writerow(row)

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

    # Final summary
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
    print(f"Mean RMSE omega:  {summary['mean_rmse_omega']:.6f} +/- {summary['std_rmse_omega']:.6f}")
    print(f"Mean RMSE theta:  {summary['mean_rmse_theta']:.6f} +/- {summary['std_rmse_theta']:.6f}")
    print(f"Mean RMSE total:  {summary['mean_rmse_total']:.6f}")
    print(f"Mean Loss:        {summary['mean_loss']:.4f} +/- {summary['std_loss']:.4f}")
    print(f"Elapsed time:     {summary['elapsed_minutes']:.1f} minutes")
    print(f"\nPer-chunk CSV: {csv_path}")
    print(f"Summary JSON:  {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
