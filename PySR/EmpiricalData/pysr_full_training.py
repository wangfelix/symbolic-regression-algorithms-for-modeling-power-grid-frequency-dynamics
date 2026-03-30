"""
PySR Per-Chunk Training – Full Dataset
========================================
Trains PySR separately on each valid 5-min chunk over the full dataset,
extracts polynomial coefficients from the best equation, and computes
ODE simulation RMSE + residual std for omega and theta.

Hyperparameters from tuning:
  sigma                 = 15
  tournament_selection_n= 10
  parsimony             = 0.0
  ncycles_per_iteration = 300
  population_size       = 50
  populations           = 50

Features:  theta, omega  (x0=theta, x1=omega)
Target:    d_omega_dt
Operators: +, -, * only  (polynomial terms)

Output CSV columns per chunk:
  chunk_id, t_start, n_points, equation, loss, complexity,
  const, omega_coef, theta_coef, omega_theta_coef,
  omega2_coef, theta2_coef, omega3_coef,
  theta2_omega_coef, theta_omega2_coef, theta3_coef,
  rmse_omega, rmse_omega_std,
  rmse_theta, rmse_theta_std,
  sim_ok
"""

import os
import math
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import sympy as sp
from scipy.ndimage import gaussian_filter1d
from pysr import PySRRegressor
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH   = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/South_Korea_2024-08-15_2025-08-31_1s.pkl"
OUT_DIR     = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/5minChunks/full_run/"
CHUNK_SIZE  = 300       # 5 min * 60 sec * 1 Hz
F_REF       = 60.0      # nominal frequency [Hz]
DT          = 1.0       # sampling interval [s]

# Best hyperparameters from tuning
SIGMA                  = 15
NITERATIONS            = 100
TOURNAMENT_SELECTION_N = 30
PARSIMONY              = 0.1
NCYCLES_PER_ITERATION  = 100
POPULATION_SIZE        = 100
POPULATIONS            = 50
MAXSIZE                = 12

# ============================================================
# Polynomial term definitions
# x0=theta, x1=omega
# ============================================================
THETA, OMEGA = sp.symbols("theta omega")
TERM_DEFS = [
    ("const",             sp.Integer(1)       ),
    ("omega_coef",        OMEGA               ),
    ("theta_coef",        THETA               ),
    ("omega_theta_coef",  OMEGA * THETA       ),
    ("omega2_coef",       OMEGA**2            ),
    ("theta2_coef",       THETA**2            ),
    ("omega3_coef",       OMEGA**3            ),
    ("theta2_omega_coef", THETA**2 * OMEGA    ),
    ("theta_omega2_coef", THETA * OMEGA**2    ),
    ("theta3_coef",       THETA**3            ),
]

# ============================================================
# Data loading and chunking
# ============================================================

def load_data(data_path: str, limit_interpolation: int = 10) -> pd.DataFrame:
    """Load pickled frequency DataFrame and interpolate short gaps."""
    print(f"Loading data from {data_path} ...")
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
    print("Extracting valid 5-min chunks from full dataset...")

    if "QI" in data.columns:
        data_filtered = data[(data["QI"] == 0) & data["freq"].notna()].dropna(subset=["freq"])
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
# Preprocessing
# ============================================================

def prepare_chunk(chunk_df: pd.DataFrame, sigma: int = SIGMA) -> tuple:
    """
    Compute omega, theta, d_omega_dt from raw frequency chunk.

    omega = (freq - 60.0) * 2*pi   [rad/s]   (Equation 6)
    theta = cumsum(omega) * dt     [rad]      (Equation 7)
    d_omega_dt = gradient(omega)   [rad/s^2]  (Equation 8)

    Returns (theta, omega, t_numeric, d_omega_dt)
    """
    freq_values = chunk_df["freq"].values
    omega_raw   = (freq_values - F_REF) * 2 * np.pi

    # Gaussian smoothing with tuned sigma
    omega = gaussian_filter1d(omega_raw.astype(float), sigma=sigma) \
            if sigma > 0 else omega_raw.astype(float)

    theta      = np.cumsum(omega) * DT
    d_omega_dt = np.gradient(omega, DT)
    t_numeric  = np.arange(len(omega)) * DT

    return theta, omega, t_numeric, d_omega_dt


# ============================================================
# Coefficient extraction
# ============================================================

def extract_coefficients(eq_str: str) -> dict:
    """
    Parse PySR equation string (x0=theta, x1=omega) and extract
    coefficients for each polynomial term. Returns NaN if term absent.
    """
    eq_sym = eq_str.replace("x0", "theta").replace("x1", "omega")
    result = {name: float("nan") for name, _ in TERM_DEFS}
    try:
        expr = sp.expand(sp.sympify(eq_sym))
        for name, term_expr in TERM_DEFS:
            c = expr.coeff(term_expr) if term_expr != sp.Integer(1) \
                else expr.as_coeff_add()[0]
            result[name] = float(c) if c != 0 else float("nan")
    except Exception:
        pass
    return result


# ============================================================
# ODE simulation
# ============================================================

_SAFE_GLOBALS = {
    "__builtins__": {},
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "abs": abs,      "pi":  math.pi,
}


def pysr_expr_to_func(eq_str):
    """Convert PySR equation string to callable. x0=theta, x1=omega."""
    code = f"lambda x0, x1: {eq_str}"
    return eval(code, _SAFE_GLOBALS)


def simulate_chunk(t_arr, theta0, omega0, dyn_func, max_seconds=10):
    import time
    deadline = time.time() + max_seconds

    def rhs(t, y):
        if time.time() > deadline:
            raise TimeoutError()
        theta, omega = y
        try:
            dw = float(dyn_func(theta, omega))
        except Exception:
            dw = 0.0
        return [omega, dw if math.isfinite(dw) else 0.0]

    try:
        sol = solve_ivp(
            rhs,
            t_span=(t_arr[0], t_arr[-1]),
            y0=[theta0, omega0],
            t_eval=t_arr,
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
            max_step=(t_arr[-1] - t_arr[0]) / 50,
        )
        if sol.success and sol.y.shape[1] == len(t_arr):
            return sol.y[0], sol.y[1]
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)


def compute_rmse_and_std(pred, true):
    """Returns (rmse, std_of_residuals). Both NaN if simulation failed."""
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    residuals = pred[mask] - true[mask]
    rmse = float(np.sqrt(np.mean(residuals**2)))
    std  = float(np.std(residuals))
    return rmse, std


# ============================================================
# PySR model factory
# ============================================================

def make_model(chunk_idx: int, out_dir: str) -> PySRRegressor:
    return PySRRegressor(
        niterations            = NITERATIONS,
        populations            = POPULATIONS,
        population_size        = POPULATION_SIZE,
        tournament_selection_n = TOURNAMENT_SELECTION_N,
        ncycles_per_iteration  = NCYCLES_PER_ITERATION,
        parsimony              = PARSIMONY,
        binary_operators       = ["+", "-", "*"],
        unary_operators        = [],
        nested_constraints     = {"*": {"*": 1}},
        maxsize                = MAXSIZE,
        variable_names         = ["theta", "omega"],
        model_selection        = "best",
        turbo                  = True,
        bumper                 = True,
        verbosity              = 0,
        precision              = 64,
        temp_equation_file     = False,
        output_directory       = os.path.join(out_dir, f"chunk_{chunk_idx:05d}"),
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PySR full-dataset per-chunk training"
    )
    parser.add_argument("--data",        default=DATA_PATH,
                        help="Path to pickled frequency dataset")
    parser.add_argument("--out",         default=OUT_DIR,
                        help="Output directory for results and HoF files")
    parser.add_argument("--niter",       type=int, default=NITERATIONS,
                        help=f"PySR iterations per chunk (default: {NITERATIONS})")
    parser.add_argument("--start-chunk", type=int, default=0,
                        help="First chunk index to process (for SLURM array)")
    parser.add_argument("--end-chunk",   type=int, default=None,
                        help="Last chunk index (exclusive). Default: all chunks.")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Process only 2 chunks for sanity check")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Load data
    data = load_data(args.data)

    # 2. Get all valid chunks
    all_chunks = get_valid_chunks(data)
    n_total    = len(all_chunks)

    # 3. Determine range to process
    start = args.start_chunk
    end   = args.end_chunk if args.end_chunk is not None else n_total
    if args.dry_run:
        end = min(start + 2, n_total)
        print("DRY RUN: processing only 2 chunks.")

    chunks_to_process = all_chunks[start:end]
    print(f"Processing chunks {start} to {end - 1}  ({len(chunks_to_process)} chunks)\n")

    # 4. Output CSV path – incremental saves go here
    out_csv = os.path.join(args.out, f"results_chunks_{start}_{end}.csv")

    results = []

    for i, (chunk_start, chunk_df) in enumerate(chunks_to_process):
        chunk_idx = start + i
        print(f"[Chunk {chunk_idx + 1}/{end}  t_start={chunk_start}] Training PySR ...")

        # Preprocessing with tuned sigma
        theta, omega, t_arr, d_omega_dt = prepare_chunk(chunk_df, sigma=SIGMA)

        X = np.stack([theta, omega], axis=1)
        y = d_omega_dt

        # Train
        try:
            model = make_model(chunk_idx, args.out)
            model.fit(X, y)

            hof      = model.equations_
            best_row = hof.loc[hof["loss"].idxmin()]
            eq_str   = str(best_row["equation"])
            loss     = float(best_row["loss"])
            complexity = int(best_row["complexity"])

        except Exception as e:
            print(f"  ERROR during training: {e}")
            row = {
                "chunk_id": chunk_idx, "t_start": str(chunk_start),
                "n_points": len(t_arr), "equation": "ERROR",
                "loss": float("nan"), "complexity": float("nan"), "sim_ok": False,
                "rmse_omega": float("nan"), "rmse_omega_std": float("nan"),
                "rmse_theta": float("nan"), "rmse_theta_std": float("nan"),
            }
            for name, _ in TERM_DEFS:
                row[name] = float("nan")
            results.append(row)
            pd.DataFrame(results).to_csv(out_csv, index=False)
            continue

        print(f"  Best eq (complexity={complexity}, loss={loss:.4e}): {eq_str}")

        # Extract polynomial coefficients
        coeffs = extract_coefficients(eq_str)

        # ODE simulation
        dyn_func              = pysr_expr_to_func(eq_str)
        theta_sim, omega_sim  = simulate_chunk(t_arr, theta[0], omega[0], dyn_func)
        rmse_omega, std_omega = compute_rmse_and_std(omega_sim, omega)
        rmse_theta, std_theta = compute_rmse_and_std(theta_sim, theta)
        sim_ok                = math.isfinite(rmse_omega)

        print(f"  RMSE omega={rmse_omega:.3e} (std={std_omega:.3e})  "
              f"RMSE theta={rmse_theta:.3e} (std={std_theta:.3e})  "
              f"sim_ok={sim_ok}")

        row = {
            "chunk_id":       chunk_idx,
            "t_start":        str(chunk_start),
            "n_points":       len(t_arr),
            "equation":       eq_str,
            "loss":           loss,
            "complexity":     complexity,
            "sim_ok":         sim_ok,
            "rmse_omega":     rmse_omega,
            "rmse_omega_std": std_omega,
            "rmse_theta":     rmse_theta,
            "rmse_theta_std": std_theta,
        }
        row.update(coeffs)
        results.append(row)

        # Save incrementally after every chunk – nothing lost on HPC timeout
        pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"\nDone. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
