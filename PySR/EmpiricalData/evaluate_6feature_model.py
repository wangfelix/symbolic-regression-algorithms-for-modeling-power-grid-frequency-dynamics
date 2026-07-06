"""
Evaluate 6-Feature South Korean Grid Model
============================================
Reads coefficients from results_recomputed_coefs.csv, reconstructs a
reduced ODE model using only 6 terms:

    d(omega)/dt = c0 + c1*theta + c2*omega + c3*omega*theta
                  + c4*omega^2*theta + c5*omega^3

Then forward-simulates each chunk against the Gaussian-smoothed
(sigma=15) empirical data and reports the mean RMSE for omega.

Usage:
    python evaluate_6feature_model.py [--data PATH] [--csv PATH]
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "results_recomputed_coefs.csv")

# Data file — parquet in the dataset dir (dsr/dataset/)
_DATA_BASE = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
_PARQUET   = os.path.join(_DATA_BASE, "South_Korea_2024-08-15_2025-08-31_1s.parquet")
DATA_PATH  = _PARQUET

CHUNK_SIZE = 300        # 5 min * 60 s * 1 Hz
F_REF      = 60.0       # nominal frequency [Hz]
DT         = 1.0        # sampling interval [s]
SIGMA      = 15         # Gaussian smoothing sigma

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4

# The 6 coefficient columns we keep (CSV column names)
SELECTED_COEF_COLS = [
    "const",              # c0  (constant)
    "theta_coef",         # c1  (theta)
    "omega_coef",         # c2  (omega)
    "omega_theta_coef",   # c3  (omega * theta)
    "theta_omega2_coef",  # c4  (omega^2 * theta)
    "omega3_coef",        # c5  (omega^3)
]


# ============================================================
# Data loading & chunking  (mirrored from pysr_full_training.py)
# ============================================================

def load_data(data_path: str, limit_interpolation: int = 10) -> pd.DataFrame:
    """Load frequency DataFrame (parquet or pickle) and interpolate short gaps."""
    print(f"Loading data from {data_path} ...")
    if data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path)
    else:
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


def get_valid_chunks(data: pd.DataFrame) -> list:
    """
    Return all valid 5-min chunks from the full dataset.
    A valid chunk has exactly 300 samples with no missing data.
    """
    print("Extracting valid 5-min chunks from full dataset ...")

    if "QI" in data.columns:
        data_filtered = data[(data["QI"] == 0) & data["freq"].notna()].dropna(
            subset=["freq"]
        )
    else:
        data_filtered = data[data["freq"].notna()].dropna(subset=["freq"])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor("5min"))

    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == CHUNK_SIZE:
            valid_chunks.append((chunk_start, group))

    print(f"Found {len(valid_chunks)} valid chunks.")
    return valid_chunks


# ============================================================
# Preprocessing  (same as pysr_full_training.py)
# ============================================================

def prepare_chunk(chunk_df: pd.DataFrame, sigma: int = SIGMA) -> tuple:
    """
    Compute omega, theta, t_numeric from raw frequency chunk.

    omega = (freq - 60.0) * 2*pi     [rad/s]
    theta = cumsum(omega) * dt        [rad]
    """
    freq_values = chunk_df["freq"].values
    omega_raw = (freq_values - F_REF) * 2 * np.pi

    # Gaussian smoothing
    omega = (
        gaussian_filter1d(omega_raw.astype(float), sigma=sigma)
        if sigma > 0
        else omega_raw.astype(float)
    )

    theta     = np.cumsum(omega) * DT
    t_numeric = np.arange(len(omega)) * DT

    return theta, omega, t_numeric


# ============================================================
# ODE simulation using the 7-term model
# ============================================================

def make_6term_rhs(c0, c1, c2, c3, c4, c5):
    """
    Build RHS function for the 6-term model:
        d(omega)/dt = c0 + c1*theta + c2*omega
                      + c3*omega*theta + c4*omega^2*theta
                      + c5*omega^3
    """
    def rhs(t, y):
        theta, omega = y
        dw = (c0
              + c1 * theta
              + c2 * omega
              + c3 * omega * theta
              + c4 * omega**2 * theta
              + c5 * omega**3)
        return [omega, dw if math.isfinite(dw) else 0.0]
    return rhs


def simulate_chunk(t_arr, theta0, omega0, rhs_func):
    """Forward-simulate the ODE using odeint and return (theta_sim, omega_sim)."""
    def rhs_odeint(y, t):
        return rhs_func(t, y)

    try:
        sol = odeint(rhs_odeint, [theta0, omega0], t_arr, full_output=False)
        if sol.shape[0] == len(t_arr):
            return sol[:, 0], sol[:, 1]
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)


def compute_rmse(pred, true):
    """RMSE between predicted and true arrays, ignoring NaN."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 6-feature reduced model via forward simulation"
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help="Path to frequency dataset (parquet or pickle)"
    )
    parser.add_argument(
        "--csv", default=CSV_PATH,
        help="Path to results_recomputed_coefs.csv"
    )
    args = parser.parse_args()

    # ----------------------------------------------------------
    # 1. Load coefficient CSV
    # ----------------------------------------------------------
    print(f"Reading coefficients from {args.csv} ...")
    df = pd.read_csv(args.csv)
    n_total = len(df)
    print(f"  Total rows in CSV: {n_total}")

    # Keep only rows where simulation was OK
    df = df[df["sim_ok"] == True].copy()
    print(f"  Rows with sim_ok=True: {len(df)}")

    # Fill NaN coefficients with 0 for the selected columns
    for col in SELECTED_COEF_COLS:
        df[col] = df[col].fillna(0.0)

    # ----------------------------------------------------------
    # 2. Load empirical data and build chunk index
    # ----------------------------------------------------------
    data = load_data(args.data)
    all_chunks = get_valid_chunks(data)

    # Build a lookup: chunk_start -> chunk_df
    chunk_lookup = {}
    for idx, (chunk_start, chunk_df) in enumerate(all_chunks):
        chunk_lookup[idx] = (chunk_start, chunk_df)

    # Also build by t_start string for matching
    tstart_to_idx = {}
    for idx, (chunk_start, _) in enumerate(all_chunks):
        tstart_to_idx[str(chunk_start)] = idx

    # ----------------------------------------------------------
    # 3. Forward-simulate each chunk
    # ----------------------------------------------------------
    rmse_values  = []
    n_stable     = 0
    n_divergent  = 0
    n_sim_fail   = 0   # solver failed / NaN output

    for row_i, row in df.iterrows():
        chunk_id = int(row["chunk_id"])
        t_start  = str(row["t_start"])

        # Match chunk by chunk_id (= index into all_chunks)
        if chunk_id not in chunk_lookup:
            # Try matching by t_start
            if t_start in tstart_to_idx:
                chunk_id = tstart_to_idx[t_start]
            else:
                n_sim_fail += 1
                continue

        chunk_start, chunk_df = chunk_lookup[chunk_id]

        # Preprocess
        theta, omega, t_arr = prepare_chunk(chunk_df, sigma=SIGMA)

        # Extract the 6 coefficients
        c0 = float(row["const"])
        c1 = float(row["theta_coef"])
        c2 = float(row["omega_coef"])
        c3 = float(row["omega_theta_coef"])
        c4 = float(row["theta_omega2_coef"])
        c5 = float(row["omega3_coef"])

        # Build RHS and simulate
        rhs_func = make_6term_rhs(c0, c1, c2, c3, c4, c5)
        theta_sim, omega_sim = simulate_chunk(
            t_arr, theta[0], omega[0], rhs_func
        )

        # Check for solver failure (all-NaN output)
        if np.all(np.isnan(omega_sim)):
            n_sim_fail += 1
            continue

        # ── Divergence check ──────────────────────────────────
        # A chunk is divergent/unstable if ANY |omega_sim| > threshold
        max_abs_omega = np.nanmax(np.abs(omega_sim))
        if max_abs_omega > OMEGA_DIVERGENCE_THRESHOLD:
            n_divergent += 1
            continue

        # ── Stable chunk → compute RMSE ───────────────────────
        rmse_omega = compute_rmse(omega_sim, omega)
        rmse_values.append(rmse_omega)
        n_stable += 1

        n_processed = n_stable + n_divergent + n_sim_fail
        if n_processed % 500 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_sim_fail} sim-fail) ...")

    # ----------------------------------------------------------
    # 4. Report results
    # ----------------------------------------------------------
    n_processed = n_stable + n_divergent + n_sim_fail
    rmse_arr = np.array(rmse_values)

    print("\n" + "=" * 60)
    print("6-Feature Model Evaluation Results")
    print("=" * 60)
    print(f"Features:  const, theta, omega, omega*theta, "
          f"omega^2*theta, omega^3")
    print(f"Smoothing: sigma={SIGMA}")
    print(f"Divergence threshold: |omega_sim| > {OMEGA_DIVERGENCE_THRESHOLD}")
    print()
    print(f"Total chunks processed: {n_processed}")
    print(f"  Stable simulations:                {n_stable}")
    print(f"  Divergent/unstable (|ω|>{OMEGA_DIVERGENCE_THRESHOLD}):  {n_divergent}")
    print(f"  Solver failures (NaN):             {n_sim_fail}")
    print()
    if len(rmse_arr) > 0:
        print(f"Forward-Simulated RMSE (omega) — stable chunks only:")
        print(f"  Mean:   {np.mean(rmse_arr):.6e}")
        print(f"  Std:    {np.std(rmse_arr):.6e}")
        print(f"  Median: {np.median(rmse_arr):.6e}")
        print(f"  Min:    {np.min(rmse_arr):.6e}")
        print(f"  Max:    {np.max(rmse_arr):.6e}")
        print(f"  25th %%: {np.percentile(rmse_arr, 25):.6e}")
        print(f"  75th %%: {np.percentile(rmse_arr, 75):.6e}")
    else:
        print("  No stable chunks found!")
    print("=" * 60)


if __name__ == "__main__":
    main()
