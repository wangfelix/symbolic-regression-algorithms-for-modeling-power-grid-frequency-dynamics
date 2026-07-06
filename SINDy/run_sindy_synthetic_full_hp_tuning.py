"""
SINDy Hyperparameter Tuning on Full Synthetic Dataset (No Chunking)

Grid search over sigma, degree, and STLSQ threshold using 1 hour
of the synthetic dataset with Wiener noise as a single time series.

Designed for SLURM array jobs: 1 combo per task.

Usage:
    python run_sindy_synthetic_full_hp_tuning.py --combo-index 0
    python run_sindy_synthetic_full_hp_tuning.py  # all combos sequentially
"""
import os
import sys
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import pysindy as ps
import argparse
import json
import datetime
import time
import itertools
import logging

# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================
HYPERPARAMETER_SPACE = {
    "sigma": [15, 30, 60],
    "degree": [1, 2, 3],
    "threshold": [1e-10, 1e-5, 1e-3, 1e-2, 1e-1],
}

# Total: 3 * 3 * 5 = 45 combinations

DT = 1.0
N_SAMPLES = 3_600        # 1 hour (3600 s)

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
# Helpers
# =============================================================================

def load_full_dataset(data_path, n_samples):
    """Load the first n_samples from the synthetic dataset."""
    log.info(f"Loading data from {data_path}...")
    with np.load(data_path) as data:
        omega = data['omega'][:n_samples]
        theta = data['theta'][:n_samples]
    log.info(f"  Loaded {len(omega)} samples ({len(omega)/86400:.1f} days)")
    return omega, theta


def generate_all_combos(space):
    keys = list(space.keys())
    values = list(space.values())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def evaluate_single_combo(omega_raw, theta_raw, hp):
    """Fit SINDy on the full dataset with given hyperparameters."""
    sigma = hp["sigma"]
    degree = hp["degree"]
    threshold = hp["threshold"]

    t0 = time.time()

    # Smoothing
    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
        theta = np.cumsum(omega) * DT
    else:
        omega = omega_raw.copy()
        theta = theta_raw.copy()

    t_arr = np.arange(len(omega)) * DT
    X = np.stack([theta, omega], axis=1)

    log.info(f"    omega std={omega.std():.6e}, range=[{omega.min():.6f}, {omega.max():.6f}]")

    # Fit SINDy
    library = ps.PolynomialLibrary(degree=degree)
    optimizer = ps.STLSQ(threshold=threshold)
    model = ps.SINDy(
        feature_names=["theta", "omega"],
        feature_library=library,
        optimizer=optimizer,
    )
    model.fit(X, t=DT)
    fit_time = time.time() - t0

    # Extract equations
    eqs = model.equations()
    eq_theta = eqs[0] if len(eqs) > 0 else "N/A"
    eq_omega = eqs[1] if len(eqs) > 1 else "N/A"

    # Extract coefficients
    coeffs_matrix = model.coefficients()
    omega_coeffs = coeffs_matrix[1, :].tolist() if coeffs_matrix.shape[0] > 1 else []

    if degree == 1:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega"]
    elif degree == 2:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]
    else:
        coeff_names = ["Coeff_Const", "Coeff_Theta", "Coeff_Omega",
                       "Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2",
                       "Coeff_Theta3", "Coeff_Theta2Omega",
                       "Coeff_ThetaOmega2", "Coeff_Omega3"]

    coeff_dict = {}
    for i, name in enumerate(coeff_names):
        coeff_dict[name] = omega_coeffs[i] if i < len(omega_coeffs) else 0.0

    # Forward simulation
    sim_rmse_omega = float('nan')
    sim_rmse_theta = float('nan')
    sim_status = "not_attempted"
    max_abs_omega_sim = float('nan')

    try:
        sim = model.simulate(X[0], t_arr, integrator="odeint")
        if np.any(np.isnan(sim)) or np.any(np.isinf(sim)):
            sim_status = "diverged_nan"
        else:
            max_abs_omega_sim = float(np.max(np.abs(sim[:, 1])))
            sim_rmse_omega = float(np.sqrt(np.mean((sim[:, 1] - X[:, 1]) ** 2)))
            sim_rmse_theta = float(np.sqrt(np.mean((sim[:, 0] - X[:, 0]) ** 2)))
            sim_status = "diverged" if max_abs_omega_sim > 0.4 else "stable"
    except Exception as e:
        sim_status = f"failed: {e}"

    total_time = time.time() - t0

    return {
        "eq_theta": eq_theta,
        "eq_omega": eq_omega,
        "coefficients": coeff_dict,
        "sim_rmse_omega": sim_rmse_omega,
        "sim_rmse_theta": sim_rmse_theta,
        "sim_status": sim_status,
        "max_abs_omega_sim": max_abs_omega_sim,
        "fit_time_s": fit_time,
        "total_time_s": total_time,
    }


# =============================================================================
# Ground truth comparison
# =============================================================================

def compare_ground_truth(coeff_dict, gt_path):
    """Compare recovered coefficients to ground truth."""
    if not os.path.exists(gt_path):
        return {}

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

    comparison = {}
    for col, (label, gt_val) in gt_map.items():
        if col in coeff_dict:
            rec = coeff_dict[col]
            if gt_val != 0 and not np.isnan(gt_val):
                rel_err = abs(rec - gt_val) / abs(gt_val) * 100
            else:
                rel_err = float('nan')
            comparison[col] = {
                "recovered": rec,
                "ground_truth": gt_val,
                "rel_error_pct": rel_err,
                "label": label,
            }
    return comparison


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SINDy HP tuning on full synthetic dataset")
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Specific combo index (for SLURM arrays)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run folder name")
    args = parser.parse_args()

    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "SVISE",
                             "synthetic_dataset_validation",
                             "synthetic_with_wiener.npz")
    gt_path = os.path.join(script_dir, "..", "SVISE",
                           "synthetic_dataset_validation",
                           "ground_truth_params.json")

    if not os.path.exists(data_path):
        log.error(f"Data not found: {data_path}")
        return

    # Load data once
    omega_raw, theta_raw = load_full_dataset(data_path, N_SAMPLES)

    # Generate combos
    all_combos = generate_all_combos(HYPERPARAMETER_SPACE)
    n_combos = len(all_combos)

    log.info(f"\n{'=' * 70}")
    log.info(f"SINDy HP TUNING — FULL SYNTHETIC DATASET ({N_SAMPLES} samples, {N_SAMPLES/3600:.0f}h)")
    log.info(f"{'=' * 70}")
    log.info(f"Search space: {n_combos} combinations")
    log.info(f"  sigma:     {HYPERPARAMETER_SPACE['sigma']}")
    log.info(f"  degree:    {HYPERPARAMETER_SPACE['degree']}")
    log.info(f"  threshold: {HYPERPARAMETER_SPACE['threshold']}")
    if args.combo_index is not None:
        log.info(f"Evaluating combo index: {args.combo_index}")
    log.info(f"{'=' * 70}\n")

    # Output directory
    results_base = os.path.join(script_dir, "results_sindy_synthetic_full_hp_tuning")
    if args.run_name:
        results_dir = os.path.join(results_base, args.run_name)
    else:
        results_dir = os.path.join(results_base, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Determine which combos to run
    if args.combo_index is not None:
        combo_indices = [args.combo_index]
    else:
        combo_indices = range(n_combos)

    best_rmse = float('inf')
    best_combo_idx = -1

    for combo_idx in combo_indices:
        if combo_idx >= n_combos:
            log.warning(f"Combo index {combo_idx} >= {n_combos}, skipping")
            continue

        hp = all_combos[combo_idx]
        log.info(f"\n{'=' * 70}")
        log.info(f"Combo {combo_idx}/{n_combos}: sigma={hp['sigma']}, "
                 f"degree={hp['degree']}, threshold={hp['threshold']:.1e}")
        log.info(f"{'=' * 70}")

        result = evaluate_single_combo(omega_raw, theta_raw, hp)

        # Print results
        log.info(f"  Eq omega:     {result['eq_omega']}")
        log.info(f"  Sim status:   {result['sim_status']}")
        log.info(f"  Sim RMSE w:   {result['sim_rmse_omega']:.6e}")
        log.info(f"  max |w_sim|:  {result['max_abs_omega_sim']:.6f}")
        log.info(f"  Fit time:     {result['fit_time_s']:.1f}s")

        # Print coefficients
        log.info(f"  Coefficients:")
        for name, val in result['coefficients'].items():
            log.info(f"    {name:<25} = {val:+.10e}")

        # Ground truth comparison
        gt_comp = compare_ground_truth(result['coefficients'], gt_path)
        if gt_comp:
            log.info(f"  Ground truth comparison:")
            for col, info in gt_comp.items():
                log.info(f"    {info['label']:<25} rec={info['recovered']:+.8e}  "
                         f"gt={info['ground_truth']:+.8e}  "
                         f"err={info['rel_error_pct']:.2f}%")

        # Save per-combo JSON
        combo_result = {
            "combo_index": combo_idx,
            "hyperparams": hp,
            "equations": {
                "d_theta_dt": result["eq_theta"],
                "d_omega_dt": result["eq_omega"],
            },
            "coefficients": result["coefficients"],
            "sim_rmse_omega": result["sim_rmse_omega"],
            "sim_rmse_theta": result["sim_rmse_theta"],
            "sim_status": result["sim_status"],
            "max_abs_omega_sim": result["max_abs_omega_sim"],
            "ground_truth_comparison": gt_comp,
            "fit_time_s": result["fit_time_s"],
            "total_time_s": result["total_time_s"],
            "n_samples": N_SAMPLES,
            "timestamp": timestamp,
        }

        combo_path = os.path.join(results_dir, f"combo_{combo_idx:03d}.json")
        with open(combo_path, 'w') as f:
            json.dump(combo_result, f, indent=4)
        log.info(f"  Saved: {combo_path}")

        # Track best
        if not np.isnan(result['sim_rmse_omega']) and result['sim_rmse_omega'] < best_rmse:
            best_rmse = result['sim_rmse_omega']
            best_combo_idx = combo_idx

    # Final summary
    elapsed = time.time() - start_time
    log.info(f"\n{'=' * 70}")
    log.info(f"DONE — {len(combo_indices)} combo(s) in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if best_combo_idx >= 0:
        log.info(f"Best: combo {best_combo_idx} -> RMSE w = {best_rmse:.6e}")
        log.info(f"  {all_combos[best_combo_idx]}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
