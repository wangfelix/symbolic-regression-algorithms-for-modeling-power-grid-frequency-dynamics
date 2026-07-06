"""
SVISE Analysis: Full Synthetic Dataset (No Chunking)

Trains a SVISE model on the entire synthetic time series (1 month = 2,592,000
samples at dt=1s) to recover the ground-truth equation coefficients.

Two variants:
  - sigma=0:  Train on raw noisy data (SVISE handles noise internally)
  - sigma=60: Train on Gaussian-smoothed data (separate script)

Usage:
    python run_svise_synthetic_full.py
"""
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import json
import datetime
import time
import copy
import logging
import functools

# Ensure svise is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

# =============================================================================
# Parameter Grid — edit these before running
# =============================================================================
SIGMA = 0                   # Gaussian smoothing sigma (0 = raw noisy data)
DEGREE = 3                  # Polynomial degree for SVISE
TAU = 0.01                  # Sparsity prior parameter
LR = 0.001                  # Learning rate
MEASUREMENT_NOISE = 1e-05   # Observation noise
N_REPARAM_SAMPLES = 15      # Number of reparameterization samples
N_TAU = 100                 # Number of tau values
MODEL_TYPE = "integrator"   # "integrator" or "sparse"

DT = 1.0                    # Sampling interval (seconds)
N_SAMPLES = 2_592_000       # 1 month of data (30 days * 86400 s/day)
T_SCALE = 30.0              # Time scaling factor

# Training parameters
MAX_EPOCHS = 10000
PATIENCE = 300
LOG_INTERVAL = 50           # Log every N epochs

# Divergence threshold for forward simulation
OMEGA_DIVERGENCE_THRESHOLD = 0.4

# =============================================================================
# Logging setup
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# Sympy-based equation unscaling
# =============================================================================
try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor
    )
    _SYMPY_AVAILABLE = True
    _SYMPY_TRANSFORMS = (standard_transformations +
                         (implicit_multiplication_application, convert_xor))
    _THETA, _OMEGA = sympy.symbols('theta omega')
    _X0, _X1 = sympy.symbols('x0 x1')
    _GLOBAL_DICT = {
        'theta': _THETA, 'omega': _OMEGA, 'x0': _X0, 'x1': _X1,
        'Symbol': sympy.Symbol, 'Float': sympy.Float,
        'Integer': sympy.Integer, 'Add': sympy.Add,
        'Mul': sympy.Mul, 'Pow': sympy.Pow,
    }
except ImportError:
    _SYMPY_AVAILABLE = False


@functools.lru_cache(maxsize=1000)
def unscale_equation(eq_str, mean_x_tuple, std_x_tuple, t_scale, feature_idx=1):
    """Convert a scaled SVISE equation back to physical units."""
    if not _SYMPY_AVAILABLE:
        return "N/A"
    if not isinstance(eq_str, str) or not eq_str or eq_str in ["N/A", "nan"]:
        return "N/A"
    if "Error" in eq_str or "FAILED" in eq_str:
        return "N/A"
    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS,
                          global_dict=_GLOBAL_DICT)
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


def extract_coeffs_from_physical_eq(eq_str):
    """Extract polynomial coefficients from a physical equation string.

    Returns [c0, c1, c2, ..., c9] for:
        c0 + c1*theta + c2*omega + c3*theta^2 + c4*theta*omega + c5*omega^2
        + c6*theta^3 + c7*theta^2*omega + c8*theta*omega^2 + c9*omega^3
    """
    if not _SYMPY_AVAILABLE or not isinstance(eq_str, str) or eq_str == "N/A":
        return None
    if "FAILED" in eq_str or "Error" in eq_str:
        return None
    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS,
                          global_dict=_GLOBAL_DICT)
        expr = sympy.expand(expr)
        expr = expr.subs({_X0: _THETA, _X1: _OMEGA})

        c0 = float(expr.subs({_THETA: 0, _OMEGA: 0}))
        c1 = float(expr.coeff(_THETA, 1).subs({_OMEGA: 0}))
        c2 = float(expr.coeff(_OMEGA, 1).subs({_THETA: 0}))
        c3 = float(expr.coeff(_THETA, 2).subs({_OMEGA: 0}))
        c4 = float(expr.coeff(_THETA * _OMEGA))
        c5 = float(expr.coeff(_OMEGA, 2).subs({_THETA: 0}))
        c6 = float(expr.coeff(_THETA, 3).subs({_OMEGA: 0}))
        c7 = float(expr.coeff(_THETA**2 * _OMEGA))
        c8 = float(expr.coeff(_THETA * _OMEGA**2))
        c9 = float(expr.coeff(_OMEGA, 3).subs({_THETA: 0}))
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    except Exception:
        return None


# =============================================================================
# Data Loading
# =============================================================================

def load_full_dataset(data_path, n_samples):
    """Load the first n_samples from the synthetic dataset."""
    log.info(f"Loading data from {data_path}...")
    with np.load(data_path) as data:
        omega = data['omega'][:n_samples]
        theta = data['theta'][:n_samples]
    log.info(f"  Loaded {len(omega)} samples ({len(omega)/86400:.1f} days)")
    return omega, theta


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "synthetic_with_wiener.npz")

    if not os.path.exists(data_path):
        log.error(f"Synthetic data not found at {data_path}")
        return

    # ── Load & prepare data ────────────────────────────────────
    omega_raw, theta_raw = load_full_dataset(data_path, N_SAMPLES)

    log.info(f"Applying Gaussian smoothing (sigma={SIGMA})...")
    t0 = time.time()
    if SIGMA > 0:
        omega = gaussian_filter1d(omega_raw, sigma=SIGMA)
        theta = np.cumsum(omega) * DT
    else:
        omega = omega_raw.copy()
        theta = theta_raw.copy()
    log.info(f"  Smoothing done in {time.time()-t0:.1f}s")

    t_arr = np.arange(len(omega)) * DT
    X = np.stack([theta, omega], axis=1)

    log.info(f"  omega range: [{omega.min():.6f}, {omega.max():.6f}]")
    log.info(f"  theta range: [{theta.min():.6f}, {theta.max():.6f}]")
    log.info(f"  omega std:   {omega.std():.6e}")

    # ── Prepare tensors ────────────────────────────────────────
    train_t = torch.tensor(t_arr, dtype=torch.float32)
    train_x = torch.tensor(X, dtype=torch.float32)

    # Global scaling
    mean_x = train_x.mean(dim=0)
    std_x = train_x.std(dim=0)
    std_x[std_x < 1e-6] = 1.0

    if MODEL_TYPE == "integrator":
        mean_x[1] = 0.0
        std_x[0] = std_x[1] * T_SCALE

    train_x_scaled = (train_x - mean_x) / std_x
    train_t_scaled = train_t / T_SCALE
    t_span = (train_t_scaled[0].item(), train_t_scaled[-1].item())

    log.info(f"  Scaling: mean_x={mean_x.tolist()}, std_x={std_x.tolist()}")
    log.info(f"  t_scale={T_SCALE}, t_span={t_span}")

    # ── Build model ────────────────────────────────────────────
    log.info(f"\nBuilding SVISE model (type={MODEL_TYPE}, degree={DEGREE})...")
    d = 2
    num_meas = 2
    G = torch.eye(d)
    measurement_noise = torch.tensor([MEASUREMENT_NOISE, MEASUREMENT_NOISE])

    common_params = {
        "d": d,
        "t_span": t_span,
        "degree": DEGREE,
        "n_reparam_samples": N_REPARAM_SAMPLES,
        "G": G,
        "num_meas": num_meas,
        "measurement_noise": measurement_noise,
        "tau": TAU,
        "train_t": train_t_scaled,
        "train_x": train_x_scaled,
        "input_labels": ["theta", "omega"],
    }

    if MODEL_TYPE == "sparse":
        model = SparsePolynomialSDE(**common_params, n_tau=N_TAU)
    elif MODEL_TYPE == "integrator":
        model = SparsePolynomialIntegratorSDE(**common_params, n_tau=N_TAU)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    log.info(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")

    # ── Training ───────────────────────────────────────────────
    log.info(f"\nStarting training (max_epochs={MAX_EPOCHS}, patience={PATIENCE})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = float('inf')
    epochs_without_improvement = 0
    nan_recoveries = 0
    max_nan_recoveries = 5
    current_lr = LR
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
            log.warning(f"  Epoch {epoch}: NaN/Inf loss → recovery {nan_recoveries}/{max_nan_recoveries}, lr={current_lr:.2e}")
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

        if epoch % LOG_INTERVAL == 0 or epoch == MAX_EPOCHS - 1:
            elapsed = time.time() - start_time
            log.info(f"  Epoch {epoch:5d} | loss={current_loss:.4f} | best={best_loss:.4f} | "
                     f"no_improve={epochs_without_improvement}/{PATIENCE} | "
                     f"elapsed={elapsed/60:.1f}min")

        if epochs_without_improvement >= PATIENCE:
            stopped_epoch = epoch
            log.info(f"  Early stopping at epoch {epoch}")
            break

        stopped_epoch = epoch

    # Restore best checkpoint
    model.load_state_dict(copy.deepcopy(best_checkpoint['model']))
    model.sde_prior.resample_weights()
    log.info(f"  Restored best checkpoint from epoch {best_checkpoint['epoch']} (loss={best_checkpoint['loss']:.4f})")

    # ── Extract equations ──────────────────────────────────────
    equations = []
    try:
        equation_strings = model.sde_prior.get_feature_names()
        equations = equation_strings
    except Exception as e:
        equations = [f"Error: {e}"] * d
        log.error(f"  Failed to extract equations: {e}")

    eq_theta_scaled = equations[0] if len(equations) > 0 else "N/A"
    eq_omega_scaled = equations[1] if len(equations) > 1 else "N/A"

    log.info(f"\n{'=' * 70}")
    log.info(f"SCALED EQUATIONS")
    log.info(f"{'=' * 70}")
    log.info(f"  d(theta)/dt = {eq_theta_scaled}")
    log.info(f"  d(omega)/dt = {eq_omega_scaled}")

    # Unscale to physical units
    mean_x_tuple = tuple(mean_x.tolist())
    std_x_tuple = tuple(std_x.tolist())

    eq_omega_phys = unscale_equation(eq_omega_scaled, mean_x_tuple, std_x_tuple,
                                     T_SCALE, feature_idx=1)

    log.info(f"\n{'=' * 70}")
    log.info(f"PHYSICAL EQUATION (unscaled)")
    log.info(f"{'=' * 70}")
    log.info(f"  d(omega)/dt = {eq_omega_phys}")

    # ── Extract coefficients ───────────────────────────────────
    coeff_names = [
        "Coeff_Const", "Coeff_Theta", "Coeff_Omega",
        "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
        "Coeff_Theta3", "Coeff_Theta2Omega", "Coeff_ThetaOmega2", "Coeff_Omega3",
    ]

    coeffs = extract_coeffs_from_physical_eq(eq_omega_phys)
    coeff_dict = {}
    log.info(f"\n{'=' * 70}")
    log.info(f"OMEGA EQUATION COEFFICIENTS")
    log.info(f"{'=' * 70}")
    if coeffs:
        for i, name in enumerate(coeff_names):
            val = coeffs[i] if i < len(coeffs) else 0.0
            coeff_dict[name] = val
            log.info(f"  {name:<25} = {val:+.10e}")
    else:
        log.warning("  Could not extract coefficients from physical equation")

    # ── Ground truth comparison ────────────────────────────────
    gt_path = os.path.join(script_dir, "ground_truth_params.json")
    gt_comparison = {}
    if os.path.exists(gt_path) and coeff_dict:
        with open(gt_path, 'r') as f:
            gt = json.load(f)

        gt_c1 = gt.get("c_1", float('nan'))
        gt_c2 = gt.get("c_2", float('nan'))
        gt_dp = gt.get("Delta_P", float('nan'))

        gt_map = {
            "Coeff_Omega": ("c_1 (damping)", gt_c1),
            "Coeff_Theta": ("c_2 (coupling)", gt_c2),
            "Coeff_Const": ("Delta_P (forcing)", gt_dp),
        }

        log.info(f"\n{'=' * 70}")
        log.info(f"GROUND TRUTH COMPARISON")
        log.info(f"{'=' * 70}")
        log.info(f"  {'Coefficient':<25} {'SVISE':>15} {'Ground Truth':>15} {'Rel. Error':>12}")
        log.info(f"  {'-' * 67}")

        for col, (label, gt_val) in gt_map.items():
            if col in coeff_dict:
                rec = coeff_dict[col]
                if gt_val != 0 and not np.isnan(gt_val):
                    rel_err = abs(rec - gt_val) / abs(gt_val) * 100
                    rel_str = f"{rel_err:.2f}%"
                else:
                    rel_str = "N/A"
                log.info(f"  {label:<25} {rec:>+15.8e} {gt_val:>+15.8e} {rel_str:>12}")
                gt_comparison[col] = {
                    "svise_value": rec,
                    "ground_truth": gt_val,
                    "label": label,
                }

        # Nonlinear terms
        for name in coeff_names:
            if name not in gt_map and coeff_dict.get(name, 0) != 0:
                rec = coeff_dict[name]
                log.info(f"  {name:<25} {rec:>+15.8e} {0.0:>+15.8e} {abs(rec):.4e}")
                gt_comparison[name] = {
                    "svise_value": rec,
                    "ground_truth": 0.0,
                    "label": name,
                }

    # ── GP State estimation RMSE ───────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"STATE ESTIMATION (GP)")
    log.info(f"{'=' * 70}")
    with torch.no_grad():
        x_pred_scaled = model.marginal_sde.mean(train_t_scaled)
        x_pred = x_pred_scaled * std_x + mean_x

        mse_theta = ((x_pred[:, 0] - train_x[:, 0]) ** 2).mean()
        mse_omega = ((x_pred[:, 1] - train_x[:, 1]) ** 2).mean()

        rmse_theta = torch.sqrt(mse_theta).item()
        rmse_omega = torch.sqrt(mse_omega).item()

    log.info(f"  GP RMSE omega: {rmse_omega:.6e}")
    log.info(f"  GP RMSE theta: {rmse_theta:.6e}")

    # ── Forward simulation from physical coefficients ──────────
    log.info(f"\n{'=' * 70}")
    log.info(f"FORWARD SIMULATION")
    log.info(f"{'=' * 70}")

    sim_result = {"gp_rmse_omega": rmse_omega, "gp_rmse_theta": rmse_theta}

    if coeffs:
        log.info(f"Simulating from recovered coefficients (n={len(t_arr)} steps)...")
        t0_sim = time.time()

        c = list(coeffs) + [0.0] * (10 - len(coeffs))

        def rhs(y, t_):
            th, om = y
            dw = (c[0] + c[1]*th + c[2]*om + c[3]*th**2 + c[4]*th*om + c[5]*om**2
                  + c[6]*th**3 + c[7]*th**2*om + c[8]*th*om**2 + c[9]*om**3)
            return [om, dw if math.isfinite(dw) else 0.0]

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = odeint(rhs, [theta[0], omega[0]], t_arr, full_output=False,
                             mxstep=5000)
            sim_time = time.time() - t0_sim
            log.info(f"  Simulation done in {sim_time:.1f}s")

            if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
                log.warning("  Simulation contains NaN/Inf!")
                sim_result["status"] = "diverged_nan"
            else:
                max_abs_omega_sim = float(np.max(np.abs(sol[:, 1])))
                log.info(f"  max |omega_sim| = {max_abs_omega_sim:.6f}")

                if max_abs_omega_sim > OMEGA_DIVERGENCE_THRESHOLD:
                    log.warning(f"  DIVERGED: max |omega_sim| = {max_abs_omega_sim:.6f}")
                    sim_result["status"] = "diverged"
                else:
                    sim_result["status"] = "stable"

                rmse_sim_omega = float(np.sqrt(np.mean((sol[:, 1] - omega) ** 2)))
                rmse_sim_theta = float(np.sqrt(np.mean((sol[:, 0] - theta) ** 2)))

                log.info(f"  Forward-Sim RMSE omega: {rmse_sim_omega:.6e}")
                log.info(f"  Forward-Sim RMSE theta: {rmse_sim_theta:.6e}")

                sim_result["sim_rmse_omega"] = rmse_sim_omega
                sim_result["sim_rmse_theta"] = rmse_sim_theta
                sim_result["max_abs_omega_sim"] = max_abs_omega_sim
        except Exception as e:
            log.error(f"  Simulation failed: {e}")
            sim_result["status"] = "failed"
            sim_result["error"] = str(e)
    else:
        log.warning("  Skipping simulation — no coefficients extracted")
        sim_result["status"] = "no_coefficients"

    # ── Save results ───────────────────────────────────────────
    results_dir = os.path.join(script_dir, "results_svise_synthetic_full")
    os.makedirs(results_dir, exist_ok=True)

    elapsed = time.time() - start_time

    summary = {
        "config": {
            "sigma": SIGMA,
            "degree": DEGREE,
            "tau": TAU,
            "lr": LR,
            "measurement_noise": MEASUREMENT_NOISE,
            "n_reparam_samples": N_REPARAM_SAMPLES,
            "n_tau": N_TAU,
            "model_type": MODEL_TYPE,
            "n_samples": N_SAMPLES,
            "n_days": N_SAMPLES / 86400,
            "dt": DT,
            "t_scale": T_SCALE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "omega_divergence_threshold": OMEGA_DIVERGENCE_THRESHOLD,
            "dataset": "synthetic_with_wiener",
        },
        "training": {
            "final_loss": best_checkpoint['loss'],
            "best_epoch": best_checkpoint['epoch'],
            "stopped_epoch": stopped_epoch,
            "nan_recoveries": nan_recoveries,
        },
        "equations": {
            "scaled_theta": eq_theta_scaled,
            "scaled_omega": eq_omega_scaled,
            "physical_omega": eq_omega_phys,
        },
        "coefficients": coeff_dict,
        "forward_simulation": sim_result,
        "ground_truth_comparison": gt_comparison,
        "elapsed_seconds": elapsed,
        "timestamp": timestamp,
    }

    sigma_tag = f"sigma{SIGMA}"
    summary_path = os.path.join(results_dir, f"summary_{sigma_tag}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    log.info(f"\n{'=' * 70}")
    log.info(f"DONE — elapsed {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"{'=' * 70}")
    log.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
