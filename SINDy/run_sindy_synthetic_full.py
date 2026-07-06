"""
SINDy Analysis: Full Synthetic Dataset (No Chunking)

Fits a PySINDy model on the entire synthetic time series (1 month = 2,592,000
samples at dt=1s) to recover the ground-truth equation coefficients.

The data (with Wiener process noise) is Gaussian-smoothed before fitting.

Usage:
    python run_sindy_synthetic_full.py
"""
import os
import sys
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import pysindy as ps
import json
import datetime
import time
import logging

# =============================================================================
# Parameter Grid — edit these before running
# =============================================================================
SIGMA = 60                  # Gaussian smoothing sigma (seconds)
DEGREE = 1                  # Polynomial library degree
STLSQ_THRESHOLD = 1e-6    # Sparsity threshold for STLSQ

DT = 1.0                   # Sampling interval (seconds)
N_SAMPLES = 86_400          # 24 hours (1 day * 86400 s)

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
# Data Loading
# =============================================================================

def load_full_dataset(data_path, n_samples=None):
    """Load the synthetic dataset. If n_samples is None, load all."""
    log.info(f"Loading data from {data_path}...")
    with np.load(data_path) as data:
        if n_samples is not None:
            omega = data['omega'][:n_samples]
            theta = data['theta'][:n_samples]
        else:
            omega = data['omega'][:]
            theta = data['theta'][:]
    log.info(f"  Loaded {len(omega)} samples ({len(omega)/86400:.1f} days)")
    return omega, theta


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "SVISE",
                             "synthetic_dataset_validation",
                             "synthetic_with_wiener.npz")

    if not os.path.exists(data_path):
        log.error(f"Synthetic data not found at {data_path}")
        return

    # ── Load data ──────────────────────────────────────────────
    omega_raw, theta_raw = load_full_dataset(data_path, N_SAMPLES)

    # ── Smooth ─────────────────────────────────────────────────
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

    # ── Fit SINDy ──────────────────────────────────────────────
    log.info(f"Fitting SINDy (degree={DEGREE}, threshold={STLSQ_THRESHOLD})...")
    t0 = time.time()

    library = ps.PolynomialLibrary(degree=DEGREE)
    optimizer = ps.STLSQ(threshold=STLSQ_THRESHOLD)
    model = ps.SINDy(
        feature_names=["theta", "omega"],
        feature_library=library,
        optimizer=optimizer,
    )
    model.fit(X, t=DT)
    fit_time = time.time() - t0
    log.info(f"  SINDy fit done in {fit_time:.1f}s")

    # ── Extract equations ──────────────────────────────────────
    eqs = model.equations()
    eq_theta = eqs[0] if len(eqs) > 0 else "N/A"
    eq_omega = eqs[1] if len(eqs) > 1 else "N/A"

    log.info(f"\n{'=' * 70}")
    log.info(f"RECOVERED EQUATIONS")
    log.info(f"{'=' * 70}")
    log.info(f"  d(theta)/dt = {eq_theta}")
    log.info(f"  d(omega)/dt = {eq_omega}")

    # ── Extract coefficients ───────────────────────────────────
    coeffs_matrix = model.coefficients()
    omega_coeffs = coeffs_matrix[1, :].tolist() if coeffs_matrix.shape[0] > 1 else []

    # Map to named coefficients
    if DEGREE == 1:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega"]
    elif DEGREE == 2:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]
    else:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
                       "Coeff_Theta3", "Coeff_Theta2Omega",
                       "Coeff_ThetaOmega2", "Coeff_Omega3"]

    coeff_dict = {}
    log.info(f"\n{'=' * 70}")
    log.info(f"OMEGA EQUATION COEFFICIENTS")
    log.info(f"{'=' * 70}")
    for i, name in enumerate(coeff_names):
        val = omega_coeffs[i] if i < len(omega_coeffs) else 0.0
        coeff_dict[name] = val
        log.info(f"  {name:<25} = {val:+.10e}")

    # ── Ground truth comparison ────────────────────────────────
    gt_path = os.path.join(script_dir, "..", "SVISE",
                           "synthetic_dataset_validation",
                           "ground_truth_params.json")
    gt_comparison = {}
    if os.path.exists(gt_path):
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
        log.info(f"  {'Coefficient':<25} {'SINDy':>15} {'Ground Truth':>15} {'Rel. Error':>12}")
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
                    "sindy_value": rec,
                    "ground_truth": gt_val,
                    "label": label,
                }

        # Nonlinear terms (should be ~0)
        for name in coeff_names:
            if name not in gt_map and coeff_dict.get(name, 0) != 0:
                rec = coeff_dict[name]
                log.info(f"  {name:<25} {rec:>+15.8e} {0.0:>+15.8e} {abs(rec):.4e}")
                gt_comparison[name] = {
                    "sindy_value": rec,
                    "ground_truth": 0.0,
                    "label": name,
                }

    # ── Forward simulation ─────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info(f"FORWARD SIMULATION")
    log.info(f"{'=' * 70}")
    log.info(f"Simulating from recovered equation (n={len(t_arr)} steps)...")
    t0 = time.time()

    sim_result = {}
    try:
        sim = model.simulate(X[0], t_arr, integrator="odeint")
        sim_time = time.time() - t0
        log.info(f"  Simulation done in {sim_time:.1f}s")

        if np.any(np.isnan(sim)) or np.any(np.isinf(sim)):
            log.warning("  Simulation contains NaN/Inf!")
            sim_result["status"] = "diverged_nan"
        else:
            max_abs_omega_sim = float(np.max(np.abs(sim[:, 1])))
            log.info(f"  max |omega_sim| = {max_abs_omega_sim:.6f}")

            if max_abs_omega_sim > 0.4:
                log.warning(f"  NOTE: max |omega_sim| = {max_abs_omega_sim:.6f} > 0.4 (would be filtered in chunk analysis)")
                sim_result["status"] = "diverged"
            else:
                sim_result["status"] = "stable"

            rmse_omega = float(np.sqrt(np.mean((sim[:, 1] - X[:, 1]) ** 2)))
            rmse_theta = float(np.sqrt(np.mean((sim[:, 0] - X[:, 0]) ** 2)))

            log.info(f"  RMSE omega: {rmse_omega:.6e}")
            log.info(f"  RMSE theta: {rmse_theta:.6e}")

            sim_result["rmse_omega"] = rmse_omega
            sim_result["rmse_theta"] = rmse_theta
            sim_result["max_abs_omega_sim"] = max_abs_omega_sim
    except Exception as e:
        log.error(f"  Simulation failed: {e}")
        sim_result["status"] = "failed"
        sim_result["error"] = str(e)

    # ── Save results ───────────────────────────────────────────
    results_dir = os.path.join(script_dir, "results_sindy_synthetic_full")
    os.makedirs(results_dir, exist_ok=True)

    elapsed = time.time() - start_time

    summary = {
        "config": {
            "sigma": SIGMA,
            "degree": DEGREE,
            "stlsq_threshold": STLSQ_THRESHOLD,
            "n_samples": len(omega),
            "n_days": len(omega) / 86400,
            "dt": DT,
            "dataset": "synthetic_with_wiener",
        },
        "equations": {
            "d_theta_dt": eq_theta,
            "d_omega_dt": eq_omega,
        },
        "coefficients": coeff_dict,
        "forward_simulation": sim_result,
        "ground_truth_comparison": gt_comparison,
        "elapsed_seconds": elapsed,
        "timestamp": timestamp,
    }

    summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    log.info(f"\n{'=' * 70}")
    log.info(f"DONE — elapsed {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"{'=' * 70}")
    log.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
