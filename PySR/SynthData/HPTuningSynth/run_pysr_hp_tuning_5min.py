"""
PySR Hyperparameter Tuning – Synthetic Dataset (5-min chunks)
=============================================================
Datensatz: synthetic_data_noiseless.pkl
  Spalten: omega [rad/s], theta [rad]
  Ziel:    d_omega_dt = np.gradient(omega, dt)

Chunk-Länge : 5 min = 300 samples (dt=1s)
Validierungsset: 300 zufällige Chunks (seed=42)
HP-Suche: 60 zufällige Kombinationen (seed=42)
SLURM: --combo-index für Array-Parallelisierung
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
DT            = 1.0
CHUNK_MINUTES = 5
CHUNK_SAMPLES = CHUNK_MINUTES * 60   # 300

MAX_CHUNKS       = 300
RANDOM_SEED      = 42
N_RANDOM_CONFIGS = 60

MAX_EPOCHS      = 100
PATIENCE_EPOCHS = 20
MAXSIZE         = 12

# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "sigma"                 : [0],
    "tournament_selection_n": [10, 20, 30],
    "parsimony"             : [1e-3, 1e-2, 1e-1, 0],
    "ncycles_per_iteration" : [100, 300, 500],
    "population_size"       : [50, 75, 100],
    "populations"           : [20, 30, 50],
}

PYSR_FIXED = dict(
    binary_operators = ["+", "-", "*"],
    unary_operators  = [],
    maxsize          = MAXSIZE,
    verbosity        = 1,
    random_state     = RANDOM_SEED,
    procs            = 1,
)

# ===========================================================================
# Data loading
# ===========================================================================
def load_data(data_path: str) -> pd.DataFrame:
    """
    Load synthetic pkl dataset.
    Erwartet Spalten: omega [rad/s], theta [rad]
    """
    logger.info("Loading synthetic data from %s ...", data_path)
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=["omega", "theta"])
    if "omega" not in data.columns or "theta" not in data.columns:
        raise ValueError(f"Expected columns 'omega' and 'theta', got: {list(data.columns)}")
    logger.info("Loaded %d rows.", len(data))
    return data

# ===========================================================================
# Chunk selection
# ===========================================================================
def get_valid_chunks(data: pd.DataFrame,
                     chunk_samples: int = CHUNK_SAMPLES,
                     max_chunks: int = MAX_CHUNKS,
                     seed: int = RANDOM_SEED) -> list:
    """
    Teilt den DataFrame in nicht-überlappende Chunks der Länge chunk_samples.
    Filtert Chunks mit NaN heraus.
    Gibt zufällig max_chunks davon zurück.
    """
    logger.info("Splitting into %d-sample chunks ...", chunk_samples)
    n_total = len(data)
    n_chunks = n_total // chunk_samples

    all_chunks = []
    for i in range(n_chunks):
        chunk = data.iloc[i * chunk_samples : (i + 1) * chunk_samples].copy()
        if chunk.isnull().any().any():
            continue
        if len(chunk) != chunk_samples:
            continue
        all_chunks.append(chunk)

    logger.info("Found %d valid chunks (no NaN, exact length).", len(all_chunks))

    if len(all_chunks) == 0:
        raise ValueError("No valid chunks found.")

    if len(all_chunks) > max_chunks:
        rng = random.Random(seed)
        all_chunks = rng.sample(all_chunks, max_chunks)
        logger.info("Randomly sampled %d chunks (seed=%d).", max_chunks, seed)

    return all_chunks

# ===========================================================================
# Preprocessing
# ===========================================================================
def prepare_chunk(chunk_df: pd.DataFrame,
                  dt: float = DT,
                  sigma: float = 0) -> np.ndarray:
    """
    Bereitet einen Chunk vor:
      - omega direkt aus Spalte lesen
      - Gaussian smoothing (sigma)
      - theta direkt aus Spalte lesen (kein cumsum nötig)
      - d_omega_dt via np.gradient
    Gibt (N, 3) Array zurück: [theta, omega, d_omega_dt]
    """
    omega_raw = chunk_df["omega"].values.astype(float)
    theta     = chunk_df["theta"].values.astype(float)

    omega = gaussian_filter1d(omega_raw, sigma=sigma) if sigma > 0 else omega_raw.copy()
    d_omega_dt = np.gradient(omega, dt)

    return np.column_stack([theta, omega, d_omega_dt])

# ===========================================================================
# Hyperparameter sampling
# ===========================================================================
def sample_random_combos(space: dict,
                         n_samples: int,
                         seed: int = RANDOM_SEED) -> list:
    rng    = random.Random(seed)
    keys   = list(space.keys())
    values = list(space.values())
    total  = 1
    for v in values:
        total *= len(v)

    if n_samples >= total:
        logger.info("Using full grid (%d combos).", total)
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    sampled_indices = rng.sample(range(total), n_samples)
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
def train_pysr(X, y, hyperparams: dict) -> PySRRegressor:
    model = PySRRegressor(
        tournament_selection_n = hyperparams["tournament_selection_n"],
        parsimony              = hyperparams["parsimony"],
        ncycles_per_iteration  = hyperparams["ncycles_per_iteration"],
        niterations            = MAX_EPOCHS,
        early_stop_condition   = "f(loss, complexity) = loss < 1e-10",
        population_size        = hyperparams["population_size"],
        populations            = hyperparams["populations"],
        **PYSR_FIXED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def train_and_evaluate_chunk(chunk_df: pd.DataFrame, hyperparams: dict) -> float:
    data_matrix = prepare_chunk(chunk_df, dt=DT, sigma=hyperparams["sigma"])
    theta      = data_matrix[:, 0]
    omega      = data_matrix[:, 1]
    d_omega_dt = data_matrix[:, 2]

    X = np.column_stack([theta, omega])
    y = d_omega_dt

    try:
        model  = train_pysr(X, y, hyperparams)
        y_pred = model.predict(X)
        return float(np.sqrt(mean_squared_error(y, y_pred)))
    except Exception as exc:
        logger.warning("Chunk failed: %s", exc)
        return float("nan")


def evaluate_hyperparams(chunks: list, hyperparams: dict) -> tuple:
    rmse_values = []
    n_total = len(chunks)
    for i, chunk_df in enumerate(chunks):
        rmse = train_and_evaluate_chunk(chunk_df, hyperparams)
        rmse_values.append(rmse)
        if (i + 1) % max(1, n_total // 10) == 0:
            valid = [r for r in rmse_values if not np.isnan(r)]
            logger.info("  Chunk %d/%d | running mean RMSE = %.6f (%d valid)",
                        i + 1, n_total, np.mean(valid) if valid else float("nan"), len(valid))
    valid_rmses = [r for r in rmse_values if not np.isnan(r)]
    mean_rmse   = float(np.mean(valid_rmses)) if valid_rmses else float("nan")
    return mean_rmse, len(valid_rmses), rmse_values

# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PySR HP Tuning – Synthetic 5-min chunks"
    )
    parser.add_argument("--data-path", type=str,
                        default="/home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/synthetic_data_noiseless.pkl")
    parser.add_argument("--results-dir", type=str,
                        default="/home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/results_hp_tuning/5min")
    parser.add_argument("--max-chunks",  type=int, default=MAX_CHUNKS)
    parser.add_argument("--n-samples",   type=int, default=N_RANDOM_CONFIGS)
    parser.add_argument("--seed",        type=int, default=RANDOM_SEED)
    parser.add_argument("--combo-index", type=int, default=None,
                        help="Nur diese Kombination auswerten (0-basiert, für SLURM array)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Schnelltest: 2 Chunks, 3 Combos, 5 Epochen")
    args = parser.parse_args()

    global MAX_EPOCHS, PATIENCE_EPOCHS
    if args.dry_run:
        MAX_EPOCHS      = 5
        PATIENCE_EPOCHS = 3
        args.max_chunks = min(args.max_chunks, 2)
        args.n_samples  = min(args.n_samples, 3)
        logger.info("DRY RUN aktiv.")

    data       = load_data(args.data_path)
    chunks     = get_valid_chunks(data,
                                  chunk_samples=CHUNK_SAMPLES,
                                  max_chunks=args.max_chunks,
                                  seed=args.seed)
    all_combos = sample_random_combos(PARAM_GRID, args.n_samples, seed=args.seed)
    n_combos   = len(all_combos)
    n_chunks   = len(chunks)

    logger.info("=" * 60)
    logger.info("PySR HP TUNING  –  5-min chunks  –  SYNTHETIC")
    logger.info("Combinations : %d", n_combos)
    logger.info("Chunks/combo : %d", n_chunks)
    logger.info("Total runs   : %d", n_combos * n_chunks)
    logger.info("=" * 60)

    os.makedirs(args.results_dir, exist_ok=True)

    if args.combo_index is not None:
        csv_path = os.path.join(args.results_dir,
                                f"hp_tuning_combo_{args.combo_index:03d}.csv")
    else:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.results_dir, f"hp_tuning_{ts}.csv")

    hp_keys = list(PARAM_GRID.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Combo_Index"] + hp_keys + ["Mean_RMSE", "Num_Success", "Num_Total"])

    best_rmse = float("inf")
    best_combo = None
    best_combo_idx = -1

    for combo_idx, hyperparams in enumerate(all_combos):
        if args.combo_index is not None and combo_idx != args.combo_index:
            continue

        logger.info("Combo %d/%d: %s", combo_idx + 1, n_combos, hyperparams)
        mean_rmse, n_success, _ = evaluate_hyperparams(chunks, hyperparams)
        logger.info("=> Mean RMSE: %.6f  (%d/%d succeeded)", mean_rmse, n_success, n_chunks)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [combo_idx + 1]
            for k in hp_keys:
                v = hyperparams[k]
                row.append(f"{v:.1e}" if isinstance(v, float) else v)
            row += [f"{mean_rmse:.6f}" if not np.isnan(mean_rmse) else "nan",
                    n_success, n_chunks]
            writer.writerow(row)

        if not np.isnan(mean_rmse) and mean_rmse < best_rmse:
            best_rmse      = mean_rmse
            best_combo     = hyperparams.copy()
            best_combo_idx = combo_idx + 1

    logger.info("Ergebnisse: %s", csv_path)

    if best_combo is not None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            args.results_dir,
            f"combo_{args.combo_index:03d}.json" if args.combo_index is not None
            else f"best_hyperparams_{ts}.json"
        )
        with open(json_path, "w") as f:
            json.dump({
                "chunk_minutes"      : CHUNK_MINUTES,
                "best_hyperparams"   : best_combo,
                "mean_rmse"          : best_rmse,
                "combo_index"        : best_combo_idx,
                "n_combos_evaluated" : n_combos,
                "n_chunks"           : n_chunks,
                "seed"               : args.seed,
                "fixed_maxsize"      : MAXSIZE,
            }, f, indent=4)
        logger.info("Best HP gespeichert: %s", json_path)


if __name__ == "__main__":
    main()
