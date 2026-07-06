"""
SVISE Final Evaluation: All Active 5-Minute Chunks from Synthetic Noisy (with Wiener) Data.

Trains the SVISE model with the best hyperparameters (from tuning) on all active
synthetic chunks. Saves per-chunk: state estimation RMSE, simulation RMSE,
scaled equations, and rescaled physical equations.

Designed for SLURM array jobs.

Usage:
    # Process chunks 0-499:
    python run_svise_synthetic_noisy.py --start-chunk 0 --end-chunk 499

    # Process all chunks:
    python run_svise_synthetic_noisy.py
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import argparse
import csv
import json
import datetime
import copy
import functools

# Ensure svise is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

# =============================================================================
# BEST HYPERPARAMETERS (update after HP tuning completes)
# =============================================================================
BEST_HYPERPARAMS = {
    "model": "integrator",
    "sigma": 0,
    "degree": 3,
    "tau": 0.01,
    "lr": 0.001,
    "n_tau": 100,
    "measurement_noise": 1e-05,
    "n_reparam_samples": 15,
}
# Source: HP tuning run_SLURM_4011874 on synthetic_with_wiener, combo 5
# Sim RMSE omega=0.057660, GP RMSE omega=0.007250, loss=895.16

# Filter threshold for dead chunks
MIN_OMEGA_STD = 0.0

# Early stopping config

MAX_EPOCHS = 10000
PATIENCE = 300


# =============================================================================
# Data Loading for Synthetic Data
# =============================================================================

def load_synthetic_data(data_path):
    """Load synthetic data and chunk into 5-minute windows."""
    print(f"Loading synthetic data from {data_path}...")

    if data_path.endswith('.npz'):
        with np.load(data_path) as data:
            omega = data['omega']
            theta = data['theta']
    else:
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
            'chunk_index': i,
        })

    print(f"  Total 5-min chunks: {len(chunks)}")
    print(f"  Keeping all {len(chunks)} chunks (noisy dataset has dynamics everywhere).")
    return chunks


def prepare_synthetic_chunk(chunk, dt=1.0, sigma=0):
    """Prepare a synthetic chunk for SVISE training."""
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
# Mathematical Helpers (equation unscaling + simulation RMSE)
# =============================================================================

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    _SYMPY_AVAILABLE = True
    _SYMPY_TRANSFORMS = (standard_transformations + (implicit_multiplication_application, convert_xor))
    _THETA, _OMEGA = sympy.symbols('theta omega')
    _X0, _X1 = sympy.symbols('x0 x1')
    _GLOBAL_DICT = {
        'theta': _THETA, 'omega': _OMEGA, 'x0': _X0, 'x1': _X1,
        'Symbol': sympy.Symbol, 'Float': sympy.Float, 'Integer': sympy.Integer,
        'Add': sympy.Add, 'Mul': sympy.Mul, 'Pow': sympy.Pow,
    }
except ImportError:
    _SYMPY_AVAILABLE = False


@functools.lru_cache(maxsize=100000)
def unscale_equation(eq_str, mean_x_tuple, std_x_tuple, t_scale, feature_idx=1):
    if not _SYMPY_AVAILABLE:
        return "N/A"
    if not isinstance(eq_str, str) or not eq_str or eq_str in ["N/A", "nan"] or "Error" in eq_str:
        return "N/A"

    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS, global_dict=_GLOBAL_DICT)
        expr = expr.subs({_X0: _THETA, _X1: _OMEGA})

        theta_sub = (_THETA - mean_x_tuple[0]) / std_x_tuple[0]
        omega_sub = (_OMEGA - mean_x_tuple[1]) / std_x_tuple[1]

        expr_sub = expr.subs({_THETA: theta_sub, _OMEGA: omega_sub})
        expr_phys = expr_sub * (std_x_tuple[feature_idx] / t_scale)
        expr_expanded = sympy.expand(expr_phys)

        for a in sympy.preorder_traversal(expr_expanded):
            if isinstance(a, sympy.Float):
                expr_expanded = expr_expanded.subs(a, round(a, 6))

        return str(expr_expanded)
    except Exception:
        return "N/A"


@functools.lru_cache(maxsize=100000)
def extract_coeffs_from_eq(eq_str):
    if not _SYMPY_AVAILABLE or not isinstance(eq_str, str) or "nan" in eq_str.lower() or "error" in eq_str.lower() or eq_str == "N/A":
        return None
    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS, global_dict=_GLOBAL_DICT)
        expr = sympy.expand(expr)
        expr = expr.subs({_X0: _THETA, _X1: _OMEGA})

        c0 = float(expr.subs({_THETA: 0, _OMEGA: 0}))
        c1 = float(expr.coeff(_THETA, 1).subs({_OMEGA: 0}))
        c2 = float(expr.coeff(_OMEGA, 1).subs({_THETA: 0}))
        c3 = float(expr.coeff(_THETA, 2).subs({_OMEGA: 0}))
        c4 = float(expr.coeff(_THETA*_OMEGA))
        c5 = float(expr.coeff(_OMEGA, 2).subs({_THETA: 0}))
        c6 = float(expr.coeff(_THETA, 3).subs({_OMEGA: 0}))
        c7 = float(expr.coeff(_THETA**2 * _OMEGA))
        c8 = float(expr.coeff(_THETA * _OMEGA**2))
        c9 = float(expr.coeff(_OMEGA, 3).subs({_THETA: 0}))
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    except Exception:
        return None


def simulate_ode_rmse(eq_str, train_x, mean_x, std_x, t_scale=30.0, dt=1.0):
    c = extract_coeffs_from_eq(eq_str)
    if c is None:
        return np.nan, np.nan, np.nan

    t = np.arange(len(train_x)) * dt
    x0_scaled = (train_x[0] - mean_x) / std_x
    t_scaled = t / t_scale

    def drift(state, t_):
        th, om = state
        domega = (c[0] + c[1]*th + c[2]*om + c[3]*th**2 + c[4]*th*om + c[5]*om**2
                  + c[6]*th**3 + c[7]*th**2*om + c[8]*th*om**2 + c[9]*om**3)
        return [om, domega]

    try:
        sol_scaled = odeint(drift, x0_scaled, t_scaled, full_output=False)
        sol = sol_scaled * std_x + mean_x

        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
            return np.nan, np.nan, np.nan
        if np.max(np.abs(sol[:, 1])) > 100 * np.max(np.abs(train_x[:, 1])):
            return np.nan, np.nan, np.nan

        rmse_om = np.sqrt(np.mean((sol[:, 1] - train_x[:, 1])**2))
        rmse_th = np.sqrt(np.mean((sol[:, 0] - train_x[:, 0])**2))
        rmse_tot = np.sqrt((rmse_om**2 + rmse_th**2)/2)
        return rmse_om, rmse_th, rmse_tot
    except Exception:
        return np.nan, np.nan, np.nan


# =============================================================================
# Training a single chunk
# =============================================================================

def train_single_chunk(chunk, hp=BEST_HYPERPARAMS):
    """Train one SVISE model on one synthetic chunk.

    Returns dict with: rmse_omega, sim_rmse_omega, equations, physical equation, etc.
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

        # State estimation RMSE
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
                "mean_x": mean_x.tolist() if isinstance(mean_x, torch.Tensor) else mean_x,
                "std_x": std_x.tolist() if isinstance(std_x, torch.Tensor) else std_x,
                "t_scale": float(t_scale),
            },
            "train_x": X_np,
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
    parser = argparse.ArgumentParser(description="SVISE evaluation on all active synthetic chunks")
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
    data_path = os.path.join(script_dir, "synthetic_with_wiener.npz")
    if not os.path.exists(data_path):
        print(f"Error: Synthetic data not found at {data_path}")
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
    print(f"SVISE FINAL EVALUATION — SYNTHETIC NOISY")
    print(f"{'=' * 60}")
    print(f"Hyperparameters: {json.dumps(BEST_HYPERPARAMS, indent=2)}")
    print(f"Total active chunks: {total_chunks}")
    print(f"Processing chunk range: [{start}, {end}] ({n_chunks} chunks)")
    print(f"Early stopping: patience={PATIENCE}, max_epochs={MAX_EPOCHS}")
    print(f"{'=' * 60}\n")

    # Output directory
    results_base = os.path.join(script_dir, "results_synthetic_noisy")
    if args.run_name:
        results_dir = os.path.join(results_base, args.run_name)
    else:
        results_dir = os.path.join(results_base, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Per-chunk CSV
    csv_filename = f"chunks_{start:05d}_to_{end:05d}_{timestamp}.csv"
    csv_path = os.path.join(results_dir, csv_filename)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Active_Chunk_Index", "Original_Chunk_Index",
            "Orig_RMSE_Omega", "Orig_RMSE_Theta", "Orig_RMSE_Total",
            "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total",
            "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
            "Eq_Theta", "Eq_Omega", "Eq_Omega_Physical"
        ])

    # Process chunks
    orig_rmse_omega_list = []
    sim_rmse_omega_list = []
    loss_list = []

    for i in range(start, end + 1):
        chunk = all_chunks[i]
        orig_idx = chunk['chunk_index']
        progress = i - start + 1

        print(f"\n--- Active chunk {i} (orig #{orig_idx}) ({progress}/{n_chunks}) ---")

        result = train_single_chunk(chunk)

        eq_theta = result["equations"][0] if len(result["equations"]) > 0 else "N/A"
        eq_omega = result["equations"][1] if len(result["equations"]) > 1 else "N/A"

        eq_omega_phys = "N/A"
        sim_om, sim_th, sim_tot = np.nan, np.nan, np.nan

        if result.get("scaling_params") is not None and "train_x" in result:
            m_x = tuple(result["scaling_params"]["mean_x"])
            s_x = tuple(result["scaling_params"]["std_x"])
            ts = result["scaling_params"]["t_scale"]

            eq_omega_phys = unscale_equation(eq_omega, m_x, s_x, ts, feature_idx=1)

            sim_om, sim_th, sim_tot = simulate_ode_rmse(
                eq_omega, result["train_x"],
                np.array(m_x), np.array(s_x), ts
            )

        orig_rmse_omega_list.append(result["rmse_omega"])
        sim_rmse_omega_list.append(sim_om)
        loss_list.append(result["final_loss"])

        print(f"    RMSE omega (GP): {result['rmse_omega']:.6f} | RMSE omega (Sim): {sim_om:.6f}")
        print(f"    Loss: {result['final_loss']:.4f} | Epoch: {result['stopped_epoch']}")
        print(f"    Eq omega: {eq_omega}")
        print(f"    Eq omega Phys: {eq_omega_phys}")

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                i, orig_idx,
                f"{result['rmse_omega']:.6f}" if not np.isnan(result['rmse_omega']) else "nan",
                f"{result['rmse_theta']:.6f}" if not np.isnan(result['rmse_theta']) else "nan",
                f"{result['rmse_total']:.6f}" if not np.isnan(result['rmse_total']) else "nan",
                f"{sim_om:.6f}" if not np.isnan(sim_om) else "nan",
                f"{sim_th:.6f}" if not np.isnan(sim_th) else "nan",
                f"{sim_tot:.6f}" if not np.isnan(sim_tot) else "nan",
                f"{result['final_loss']:.4f}" if not np.isnan(result['final_loss']) else "nan",
                result['stopped_epoch'],
                result['nan_recoveries'],
                eq_theta,
                eq_omega,
                eq_omega_phys
            ])

        if progress % 50 == 0 or progress == n_chunks:
            val_orig = [r for r in orig_rmse_omega_list if not np.isnan(r)]
            val_sim = [r for r in sim_rmse_omega_list if not np.isnan(r)]
            valid_losses = [r for r in loss_list if not np.isnan(r)]

            mean_orig = np.mean(val_orig) if val_orig else float('nan')
            mean_sim = np.mean(val_sim) if val_sim else float('nan')
            mean_loss = np.mean(valid_losses) if valid_losses else float('nan')

            elapsed = (datetime.datetime.now() - start_time).total_seconds() / 60
            rate = elapsed / progress
            eta = rate * (n_chunks - progress)
            print(f"\n  [Progress {progress}/{n_chunks}] "
                  f"Mean RMSE omega (GP): {mean_orig:.6f} | (Sim): {mean_sim:.6f} | "
                  f"Mean loss: {mean_loss:.4f} | "
                  f"Valid GP: {len(val_orig)}/{progress} | Valid Sim: {len(val_sim)}/{progress} | "
                  f"Elapsed: {elapsed:.1f}min | ETA: {eta:.1f}min")

    # Final summary
    val_orig = [r for r in orig_rmse_omega_list if not np.isnan(r)]
    val_sim = [r for r in sim_rmse_omega_list if not np.isnan(r)]
    valid_losses = [r for r in loss_list if not np.isnan(r)]

    summary = {
        "hyperparams": BEST_HYPERPARAMS,
        "chunk_range": [start, end],
        "n_chunks_processed": n_chunks,
        "n_chunks_succeeded_gp": len(val_orig),
        "n_chunks_succeeded_sim": len(val_sim),
        "mean_orig_rmse_omega": float(np.mean(val_orig)) if val_orig else float('nan'),
        "std_orig_rmse_omega": float(np.std(val_orig)) if val_orig else float('nan'),
        "mean_sim_rmse_omega": float(np.mean(val_sim)) if val_sim else float('nan'),
        "std_sim_rmse_omega": float(np.std(val_sim)) if val_sim else float('nan'),
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
    print(f"EVALUATION COMPLETE — Synthetic Noisy (with Wiener) [{start}, {end}]")
    print(f"{'=' * 60}")
    print(f"Chunks processed: {n_chunks}")
    print(f"Chunks succeeded GP:  {len(val_orig)} ({100*len(val_orig)/n_chunks:.1f}%)")
    print(f"Chunks succeeded Sim: {len(val_sim)} ({100*len(val_sim)/n_chunks:.1f}%)")
    print(f"Mean GP RMSE omega:   {summary['mean_orig_rmse_omega']:.6f} +/- {summary['std_orig_rmse_omega']:.6f}")
    print(f"Mean SIM RMSE omega:  {summary['mean_sim_rmse_omega']:.6f} +/- {summary['std_sim_rmse_omega']:.6f}")
    print(f"Mean Loss:            {summary['mean_loss']:.4f} +/- {summary['std_loss']:.4f}")
    print(f"Elapsed time:         {summary['elapsed_minutes']:.1f} minutes")
    print(f"\nPer-chunk CSV: {csv_path}")
    print(f"Summary JSON:  {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
