"""
PySR Per-Chunk Training – Synthetic Dataset (5-min chunks)
==========================================================
Trainiert PySR auf jedem gültigen 5-min Chunk des synthetischen Datensatzes.

Datensatz: synthetic_data_noiseless.pkl
  Spalten: omega [rad/s], theta [rad]
  Ziel:    d_omega_dt = np.gradient(omega, dt)

Hyperparameter: aus HP-Tuning übernehmen (nach Abschluss anpassen)

Output CSV pro Chunk:
  chunk_id, t_start, n_points, equation, loss, complexity, sim_ok,
  rmse_omega, rmse_omega_std, rmse_theta, rmse_theta_std,
  const, omega_coef, theta_coef, omega_theta_coef,
  omega2_coef, theta2_coef, omega3_coef,
  theta2_omega_coef, theta_omega2_coef, theta3_coef
"""

import os
import math
import warnings
import argparse
import pickle
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
DATA_PATH  = "/home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/synthetic_data_noiseless.pkl"
OUT_DIR    = "/home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/results_synthetic_noise/5minNewParam/"
CHUNK_SIZE = 300    # 5 min * 60 sec * 1 Hz
DT         = 1.0   # sampling interval [s]

# ── Hyperparameter (nach HP-Tuning anpassen) ──────────────────────────────
SIGMA                  = 0
NITERATIONS            = 100
TOURNAMENT_SELECTION_N = 30
PARSIMONY              = 0.01
NCYCLES_PER_ITERATION  = 300
POPULATION_SIZE        = 30
POPULATIONS            = 50
MAXSIZE                = 12

# ============================================================
# Polynomial term definitions (x0=theta, x1=omega)
# ============================================================
THETA, OMEGA = sp.symbols("theta omega")
TERM_DEFS = [
    ("const",             sp.Integer(1)      ),
    ("omega_coef",        OMEGA              ),
    ("theta_coef",        THETA              ),
    ("omega_theta_coef",  OMEGA * THETA      ),
    ("omega2_coef",       OMEGA**2           ),
    ("theta2_coef",       THETA**2           ),
    ("omega3_coef",       OMEGA**3           ),
    ("theta2_omega_coef", THETA**2 * OMEGA   ),
    ("theta_omega2_coef", THETA * OMEGA**2   ),
    ("theta3_coef",       THETA**3           ),
]

# ============================================================
# Data loading
# ============================================================
def load_data(data_path: str) -> pd.DataFrame:
    print(f"Loading synthetic data from {data_path} ...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=["omega", "theta"])
    if "omega" not in data.columns or "theta" not in data.columns:
        raise ValueError(f"Expected columns 'omega' and 'theta', got: {list(data.columns)}")
    print(f"Loaded {len(data):,} rows.")
    return data

# ============================================================
# Chunk selection
# ============================================================
def get_valid_chunks(data: pd.DataFrame) -> list:
    print(f"Splitting into {CHUNK_SIZE}-sample (5-min) chunks ...")
    n_chunks   = len(data) // CHUNK_SIZE
    all_chunks = []
    for i in range(n_chunks):
        chunk = data.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE].copy()
        if not chunk.isnull().any().any() and len(chunk) == CHUNK_SIZE:
            all_chunks.append((i, chunk))
    print(f"Found {len(all_chunks)} valid chunks.")
    return all_chunks

# ============================================================
# Preprocessing
# ============================================================
def prepare_chunk(chunk_df: pd.DataFrame, sigma: int = SIGMA) -> tuple:
    """
    Liest omega und theta direkt aus dem DataFrame.
    Berechnet d_omega_dt via np.gradient.
    Returns (theta, omega, t_arr, d_omega_dt)
    """
    omega_raw  = chunk_df["omega"].values.astype(float)
    theta      = chunk_df["theta"].values.astype(float)
    omega      = gaussian_filter1d(omega_raw, sigma=sigma) if sigma > 0 else omega_raw.copy()
    d_omega_dt = np.gradient(omega, DT)
    t_arr      = np.arange(len(omega)) * DT
    return theta, omega, t_arr, d_omega_dt

# ============================================================
# Coefficient extraction
# ============================================================
def extract_coefficients(eq_str: str) -> dict:
    eq_sym = eq_str.replace("x0", "theta").replace("x1", "omega")
    result = {name: float("nan") for name, _ in TERM_DEFS}
    try:
        expr = sp.expand(sp.sympify(eq_sym))
        poly = sp.Poly(expr, THETA, OMEGA)
        present = set(poly.monoms())
        for name, term_expr in TERM_DEFS:
            if term_expr == sp.Integer(1):
                monom = (0, 0)
            else:
                monom = tuple(sp.Poly(term_expr, THETA, OMEGA).monoms()[0])
            if monom in present:
                val = float(poly.coeff_monomial(term_expr))
                result[name] = val  # auch 0.0 eintragen wenn explizit vorhanden
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
    "abs": abs, "pi": math.pi,
}

def pysr_expr_to_func(eq_str):
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
            rtol=1e-4, atol=1e-6,
            max_step=(t_arr[-1] - t_arr[0]) / 50,
        )
        if sol.success and sol.y.shape[1] == len(t_arr):
            return sol.y[0], sol.y[1]
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)

def compute_rmse_and_std(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    residuals = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(residuals**2))), float(np.std(residuals))

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
        description="PySR full training – Synthetic 5-min chunks"
    )
    parser.add_argument("--data",        default=DATA_PATH)
    parser.add_argument("--out",         default=OUT_DIR)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk",   type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Nur 2 Chunks verarbeiten")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data       = load_data(args.data)
    all_chunks = get_valid_chunks(data)
    n_total    = len(all_chunks)

    start = args.start_chunk
    end   = args.end_chunk if args.end_chunk is not None else n_total
    if args.dry_run:
        end = min(start + 2, n_total)
        print("DRY RUN: nur 2 Chunks.")

    chunks_to_process = all_chunks[start:end]
    print(f"Verarbeite Chunks {start} bis {end - 1}  ({len(chunks_to_process)} Chunks)\n")

    out_csv = os.path.join(args.out, f"results_chunks_{start}_{end}.csv")
    results = []

    for i, (chunk_idx, chunk_df) in enumerate(chunks_to_process):
        global_idx = start + i
        print(f"[Chunk {global_idx + 1}/{end}  idx={chunk_idx}] Training PySR ...")

        theta, omega, t_arr, d_omega_dt = prepare_chunk(chunk_df, sigma=SIGMA)
        X = np.stack([theta, omega], axis=1)
        y = d_omega_dt

        try:
            model    = make_model(global_idx, args.out)
            model.fit(X, y)
            hof      = model.equations_
            best_row = hof.loc[hof["loss"].idxmin()]
            eq_str   = str(best_row["equation"])
            loss     = float(best_row["loss"])
            complexity = int(best_row["complexity"])
        except Exception as e:
            print(f"  FEHLER beim Training: {e}")
            row = {
                "chunk_id": global_idx, "t_start": chunk_idx,
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

        print(f"  Beste Gleichung (complexity={complexity}, loss={loss:.4e}): {eq_str}")

        coeffs                        = extract_coefficients(eq_str)
        dyn_func                      = pysr_expr_to_func(eq_str)
        theta_sim, omega_sim          = simulate_chunk(t_arr, theta[0], omega[0], dyn_func)
        rmse_omega, std_omega         = compute_rmse_and_std(omega_sim, omega)
        rmse_theta, std_theta         = compute_rmse_and_std(theta_sim, theta)
        sim_ok                        = math.isfinite(rmse_omega)

        print(f"  RMSE omega={rmse_omega:.3e} (std={std_omega:.3e})  "
              f"RMSE theta={rmse_theta:.3e} (std={std_theta:.3e})  "
              f"sim_ok={sim_ok}")

        row = {
            "chunk_id":       global_idx,
            "t_start":        chunk_idx,
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
        pd.DataFrame(results).to_csv(out_csv, index=False)

    print(f"\nFertig. Ergebnisse: {out_csv}")


if __name__ == "__main__":
    main()
