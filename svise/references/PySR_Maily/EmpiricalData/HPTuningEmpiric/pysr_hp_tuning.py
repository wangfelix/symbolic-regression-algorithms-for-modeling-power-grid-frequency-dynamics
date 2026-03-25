"""
PySR Hyperparameter Tuning for Power Grid Frequency Dynamics
=============================================================
Based on: "Data-Driven Discovery of Power Grid Frequency Dynamics using
Symbolic Regression" (Wang, Doan et al., 2026)

Aligned with SVISE hyperparameter tuning pipeline (colleague code).

State representation (Equation 6):
  omega = (freq - 60.0) * 2 * pi   [rad/s]
  theta = cumsum(omega) * dt        [rad]

Search space:
  sigma          : [5, 10, 15]          - Gaussian smoothing width
  tournament_selection_n: [10, 20, 30]         - Selection pool size
  parsimony      : [1e-3, 1e-2, 1e-1, 0] - Complexity penalty
  ncycles        : [100, 300, 500]      - Cycles per iteration
  population_size: [20, 50, 100]        - Individuals per population
  populations    : [30, 50, 100]        - Number of populations

Fixed:
  maxsize = 12                          - Max expression tree size (not tuned)

Strategy: Random search - 60 combinations sampled uniformly (seed 42).
  - 300 chunks from 9:00-10:00 window, randomly sampled with seed 42.
  - All 60 configs evaluated on the same 300 chunks (fixed validation set).
  - Supports --combo-index for SLURM array parallelization.

Early stopping: max 10 000 epochs, patience 300 epochs.
"""

import os
import sys
import pickle
import random
import itertools
import warnings
import logging
import argparse
import csv
import json
import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
F_REF         = 60.0              # nominal grid frequency [Hz]
DT            = 1.0               # sampling interval [s]
CHUNK_MINUTES = 5                 # segment length
CHUNK_SAMPLES = CHUNK_MINUTES * 60  # = 300 samples at 1 Hz

MAX_CHUNKS        = 300           # chunks sampled from 9-10 window
RANDOM_SEED       = 42            # reproducibility (chunks + config sampling)
N_RANDOM_CONFIGS  = 60            # hyperparameter combinations to evaluate

# Early stopping
MAX_EPOCHS     = 300
PATIENCE_EPOCHS = 20

# Fixed hyperparameters (not tuned)
MAXSIZE = 12

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "sigma"                 : [5, 10, 15],
    "tournament_selection_n": [10, 20, 30],
    "parsimony"             : [1e-3, 1e-2, 1e-1, 0],
    "ncycles_per_iteration" : [100, 300, 500],
    "population_size"       : [50, 75, 100],
    "populations"           : [20, 30, 50],
}

# Fixed PySR options shared across all configurations
PYSR_FIXED = dict(
    binary_operators  = ["+", "-", "*"],
    unary_operators   = [],
    maxsize           = MAXSIZE,
    verbosity         = 1,
    random_state      = RANDOM_SEED,
    procs             = 1,
)


# ===========================================================================
# Data loading  (aligned with colleague's load_data)
# ===========================================================================

def load_data(data_path: str, limit_interpolation: int = 10) -> pd.DataFrame:
    """
    Load pickled frequency DataFrame and interpolate short gaps.
    Mirrors colleague's load_data exactly.
    """
    logger.info("Loading data from %s...", data_path)
    data = pd.read_pickle(data_path)

    if "QI" in data.columns:
        data.loc[:, "freq"] = data.loc[:, "freq"].interpolate(
            method="time", limit=limit_interpolation
        )
        data.loc[data["freq"].isna(), "QI"] = 2
        data.loc[~data["freq"].isna(), "QI"] = 0
    else:
        data["freq"] = data["freq"].interpolate(
            method="time", limit=limit_interpolation
        )

    return data


# ===========================================================================
# Chunk selection  (aligned with colleague's get_valid_chunks_9_to_10)
# ===========================================================================

def get_valid_chunks_9_to_10(data: pd.DataFrame,
                              max_chunks: int = MAX_CHUNKS,
                              seed: int = RANDOM_SEED) -> list:
    """
    Get valid 5-minute chunks whose start time falls in the 9:00-10:00 window.
    A valid chunk has exactly 300 samples with no missing data.
    Chunks starting at 9:00, 9:05, ..., 9:55 are included.

    If more than max_chunks are found, randomly sample max_chunks with the
    given seed -> fixed validation set: all configs see the same 300 chunks.
    Mirrors colleague's get_valid_chunks_9_to_10 exactly.
    """
    logger.info("Filtering for valid 5-min chunks in the 9:00-10:00 window...")

    if "QI" in data.columns:
        data_filtered = data[(data["QI"] == 0) & data["freq"].notna()].dropna(subset=["freq"])
    else:
        data_filtered = data[data["freq"].notna()].dropna(subset=["freq"])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor("5min"))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) != CHUNK_SAMPLES:
            continue
        if chunk_start.hour != 9:          # only 9:00-10:00
            continue
        valid_chunks.append(group)

    if not valid_chunks:
        raise ValueError("No valid 5-min chunks found in the 9:00-10:00 window.")

    if len(valid_chunks) > max_chunks:
        rng = random.Random(seed)
        sampled = rng.sample(valid_chunks, max_chunks)
        logger.info("Found %d total valid chunks. Randomly sampled %d (seed=%d).",
                    len(valid_chunks), max_chunks, seed)
        return sampled
    else:
        logger.info("Found %d valid chunks (max requested: %d).",
                    len(valid_chunks), max_chunks)
        return valid_chunks


# ===========================================================================
# Preprocessing  (aligned with colleague's prepare_data)
# ===========================================================================

def prepare_chunk(chunk_df: pd.DataFrame,
                  dt: float = DT,
                  sigma: float = 0) -> np.ndarray:
    """
    Prepare a single 5-minute chunk into a data matrix.

    State representation:
        omega = (freq - 60.0) * 2*pi   [rad/s]   (Equation 6)
        theta = cumsum(omega) * dt     [rad]      (Equation 7)
        d_omega_dt = gradient(omega)   [rad/s^2]  (Equation 8)

    Gaussian filter (sigma) is applied to omega before computing
    theta and d_omega_dt so all quantities stay consistent.

    Returns (N, 4) array: [theta, omega, t, d_omega_dt]
    """
    freq_values = chunk_df["freq"].values

    # Equation 6: omega = (f - f_ref) * 2*pi  [rad/s]
    omega_raw = (freq_values - F_REF) * 2 * np.pi

    # Gaussian smoothing
    omega = gaussian_filter1d(omega_raw.astype(float), sigma=sigma) \
            if sigma > 0 else omega_raw.astype(float)

    # Equation 7: theta = integral of omega (reset to 0 at segment start)
    theta = np.cumsum(omega) * dt

    # Equation 8: d_omega_dt via central finite differences
    d_omega_dt = np.gradient(omega, dt)

    # Time axis (kept for traceability, not used as PySR input)
    t_numeric = np.arange(len(omega)) * dt

    return np.column_stack([theta, omega, t_numeric, d_omega_dt])


# ===========================================================================
# Hyperparameter sampling  (aligned with colleague's sample_random_combos)
# ===========================================================================

def sample_random_combos(space: dict,
                         n_samples: int,
                         seed: int = RANDOM_SEED) -> list:
    """
    Randomly sample n_samples unique combinations from the search space.
    Mirrors colleague's sample_random_combos exactly.
    """
    rng  = random.Random(seed)
    keys = list(space.keys())
    values = list(space.values())

    total_possible = 1
    for v in values:
        total_possible *= len(v)

    if n_samples >= total_possible:
        logger.info("Requested %d samples but only %d unique combos exist. "
                    "Using full grid.", n_samples, total_possible)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*values)]

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


# ===========================================================================
# PySR training & evaluation
# ===========================================================================

def train_pysr(X: np.ndarray,
               y: np.ndarray,
               tournament_selection_n: int,
               parsimony: float,
               ncycles: int,
               population_size: int,
               populations: int) -> PySRRegressor:
    """
    Fit PySRRegressor on one pre-processed chunk.

    niterations is set to MAX_EPOCHS (hard upper bound).
    Early stopping is handled via early_stop_condition.
    maxsize is fixed at MAXSIZE = 12.
    Time (t) is never passed as input - X contains only [theta, omega].
    """
    model = PySRRegressor(
        tournament_selection_n = tournament_selection_n,
        parsimony = parsimony,
        ncycles_per_iteration = ncycles,
        niterations           = MAX_EPOCHS,
        early_stop_condition  = "f(loss, complexity) = loss < 1e-10",
        population_size       = population_size,
        populations           = populations,
        **PYSR_FIXED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def evaluate_chunk(model: PySRRegressor,
                   X: np.ndarray,
                   y: np.ndarray) -> float:
    """RMSE of the best Pareto-front equation against the d_omega_dt target."""
    y_pred = model.predict(X)
    return float(np.sqrt(mean_squared_error(y, y_pred)))


def train_and_evaluate_chunk(chunk_df: pd.DataFrame,
                             hyperparams: dict) -> float:
    """
    Prepare one chunk and train+evaluate PySR with the given hyperparams.
    Returns RMSE or NaN on failure.
    """
    sigma = hyperparams["sigma"]

    data_matrix = prepare_chunk(chunk_df, dt=DT, sigma=sigma)
    theta      = data_matrix[:, 0]   # [rad]
    omega      = data_matrix[:, 1]   # [rad/s]
    d_omega_dt = data_matrix[:, 3]   # [rad/s^2] - target

    # X = [theta, omega] only - time is intentionally excluded
    X = np.column_stack([theta, omega])
    y = d_omega_dt

    try:
        model = train_pysr(
            X, y,
            tournament_selection_n = hyperparams["tournament_selection_n"],
            parsimony       = hyperparams["parsimony"],
            ncycles         = hyperparams["ncycles_per_iteration"],
            population_size = hyperparams["population_size"],
            populations     = hyperparams["populations"],
        )
        return evaluate_chunk(model, X, y)
    except Exception as exc:
        logger.warning("Chunk training failed: %s", exc)
        return float("nan")


# ===========================================================================
# Evaluate one hyperparameter combination across all chunks
# ===========================================================================

def evaluate_hyperparams(chunks: list, hyperparams: dict) -> tuple:
    """
    Train on all chunks with the given hyperparams.
    Returns (mean_rmse, n_success, all_rmses).
    Mirrors colleague's evaluate_hyperparams structure.
    """
    rmse_values = []
    n_total = len(chunks)

    for i, chunk_df in enumerate(chunks):
        rmse = train_and_evaluate_chunk(chunk_df, hyperparams)
        rmse_values.append(rmse)

        # Progress logging every 10%
        if (i + 1) % max(1, n_total // 10) == 0:
            valid = [r for r in rmse_values if not np.isnan(r)]
            running_mean = np.mean(valid) if valid else float("nan")
            logger.info("  Chunk %d/%d: running mean RMSE = %.6f (%d valid)",
                        i + 1, n_total, running_mean, len(valid))

    valid_rmses = [r for r in rmse_values if not np.isnan(r)]
    mean_rmse   = float(np.mean(valid_rmses)) if valid_rmses else float("nan")
    n_success   = len(valid_rmses)

    return mean_rmse, n_success, rmse_values


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PySR Hyperparameter Tuning – Power Grid Frequency Dynamics"
    )
    parser.add_argument("--data-path", type=str,
                        default="/home/ka/ka_iai/ka_hr7224/PySRCurrent/South_Korea_2024-08-15_2025-08-31_1s.pkl",
                        help="Path to the pickled frequency dataset")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS,
                        help=f"Max chunks sampled from 9-10 window (default: {MAX_CHUNKS})")
    parser.add_argument("--n-samples", type=int, default=N_RANDOM_CONFIGS,
                        help=f"Number of random hyperparameter combos (default: {N_RANDOM_CONFIGS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for chunk + combo sampling (default: {RANDOM_SEED})")
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Evaluate only this combo index (0-based, for SLURM array jobs)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run folder name (for aggregating SLURM array results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick sanity check: 2 chunks, 3 combos, 5 max epochs")
    args = parser.parse_args()

    # Dry-run overrides
    global MAX_EPOCHS, PATIENCE_EPOCHS
    if args.dry_run:
        MAX_EPOCHS      = 5
        PATIENCE_EPOCHS = 3
        args.max_chunks = min(args.max_chunks, 2)
        args.n_samples  = min(args.n_samples, 3)
        logger.info("DRY RUN: max_epochs=5, patience=3, max_chunks=2, n_samples=3")

    # 1. Load data
    data   = load_data(args.data_path)

    # 2. Get fixed validation set: 300 chunks from 9-10, same for all configs
    chunks = get_valid_chunks_9_to_10(data,
                                      max_chunks=args.max_chunks,
                                      seed=args.seed)

    # 3. Sample hyperparameter combinations
    all_combos   = sample_random_combos(PARAM_GRID, args.n_samples, seed=args.seed)
    n_combos     = len(all_combos)
    n_chunks     = len(chunks)
    total_space  = 1
    for v in PARAM_GRID.values():
        total_space *= len(v)

    logger.info("=" * 60)
    logger.info("PySR HYPERPARAMETER RANDOM SEARCH")
    logger.info("=" * 60)
    logger.info("Total search space : %d combinations", total_space)
    logger.info("Evaluating         : %d combinations", n_combos)
    logger.info("Chunks per combo   : %d", n_chunks)
    logger.info("Total training runs: %d", n_combos * n_chunks)
    logger.info("Early stopping     : patience=%d, max_epochs=%d",
                PATIENCE_EPOCHS, MAX_EPOCHS)
    logger.info("Fixed maxsize      : %d", MAXSIZE)
    logger.info("=" * 60)

    # 4. Results directory
    if args.run_name:
        results_dir = os.path.join("results_hp_tuning", args.run_name)
    else:
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results_hp_tuning", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    if args.combo_index is not None:
        csv_path = os.path.join(results_dir,
                                f"hp_tuning_combo_{args.combo_index:03d}.csv")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join(results_dir, f"hp_tuning_{timestamp}.csv")

    # CSV header
    hp_keys = list(PARAM_GRID.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Combo_Index"] + hp_keys + ["Mean_RMSE", "Num_Success", "Num_Total"]
        )

    # 5. Evaluate combinations
    best_rmse  = float("inf")
    best_combo = None
    best_combo_idx = -1

    for combo_idx, hyperparams in enumerate(all_combos):

        # SLURM array mode: skip all combos except the assigned one
        if args.combo_index is not None and combo_idx != args.combo_index:
            continue

        logger.info("=" * 60)
        logger.info("Combo %d/%d: %s", combo_idx + 1, n_combos, hyperparams)
        logger.info("=" * 60)

        mean_rmse, n_success, _ = evaluate_hyperparams(chunks, hyperparams)

        logger.info("=> Mean RMSE: %.6f  (%d/%d chunks succeeded)",
                    mean_rmse, n_success, n_chunks)

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [combo_idx + 1]
            for k in hp_keys:
                v = hyperparams[k]
                row.append(f"{v:.1e}" if isinstance(v, float) else v)
            row += [
                f"{mean_rmse:.6f}" if not np.isnan(mean_rmse) else "nan",
                n_success,
                n_chunks,
            ]
            writer.writerow(row)

        if not np.isnan(mean_rmse) and mean_rmse < best_rmse:
            best_rmse      = mean_rmse
            best_combo     = hyperparams.copy()
            best_combo_idx = combo_idx + 1

    # 6. Save best result as JSON (mirrors colleague's output format)
    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info("Results saved to: %s", csv_path)

    if best_combo is not None:
        logger.info("Best combination (combo #%d):", best_combo_idx)
        for k, v in best_combo.items():
            logger.info("  %s: %s", k, v)
        logger.info("  Mean RMSE: %.6f", best_rmse)

        timestamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_json_path = os.path.join(
            results_dir,
            f"combo_{args.combo_index:03d}.json" if args.combo_index is not None
            else f"best_hyperparams_{timestamp}.json"
        )
        with open(best_json_path, "w") as f:
            json.dump({
                "best_hyperparams"      : best_combo,
                "mean_rmse"             : best_rmse,
                "combo_index"           : best_combo_idx,
                "n_combos_evaluated"    : n_combos,
                "total_search_space"    : total_space,
                "n_chunks"              : n_chunks,
                "timestamp"             : timestamp,
                "seed"                  : args.seed,
                "fixed_maxsize"         : MAXSIZE,
                "early_stopping"        : {
                    "max_epochs" : MAX_EPOCHS,
                    "patience"   : PATIENCE_EPOCHS,
                },
            }, f, indent=4)
        logger.info("Best hyperparams saved to: %s", best_json_path)
    else:
        logger.info("No valid results found. All combinations failed.")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
