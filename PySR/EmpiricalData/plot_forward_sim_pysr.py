"""
Plot forward-simulated omega (from PySR-recovered equation) vs empirical omega.

Reads the equation from a PySR results CSV, re-simulates the ODE from initial
conditions, and plots against the empirical + smoothed data for visual comparison.

If --train-missing is passed and a requested chunk is not in the CSV, PySR is
trained on-the-fly for that chunk using the tuned hyperparameters.

Usage:
    python plot_forward_sim_pysr.py --best 50
    python plot_forward_sim_pysr.py --chunks 1234 5678
    python plot_forward_sim_pysr.py --chunks 98237 --train-missing
    python plot_forward_sim_pysr.py --best 20 --csv /path/to/results.csv
"""
import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

SIGMA = 15
DT = 1.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "results_all_combined.csv")

# ── PySR hyperparameters (from tuning) ──────────────────────────────────────
NITERATIONS            = 100
TOURNAMENT_SELECTION_N = 30
PARSIMONY              = 0.1
NCYCLES_PER_ITERATION  = 100
POPULATION_SIZE        = 100
POPULATIONS            = 50
MAXSIZE                = 12


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(data_path, limit_interpolation=10):
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)
    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)
    return data


def get_all_valid_chunks(data):
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))
    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == 300:
            valid_chunks.append((chunk_start, group))
    return valid_chunks


def prepare_chunk(chunk_df, dt=DT):
    """Return raw omega, smoothed omega (sigma=15), theta (from smoothed), and time."""
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi
    omega_smooth = gaussian_filter1d(omega_raw, sigma=SIGMA)
    theta_smooth = np.cumsum(omega_smooth) * dt
    t = np.arange(len(omega_raw)) * dt
    return t, omega_raw, omega_smooth, theta_smooth


# ── Equation Parsing & Simulation ────────────────────────────────────────────

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


def simulate_chunk(t_arr, theta0, omega0, dyn_func):
    """Simulate the ODE: dtheta/dt = omega, domega/dt = f(theta, omega)."""
    def rhs(t, y):
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
            return sol.y[0], sol.y[1]  # theta_sim, omega_sim
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)


# ── On-the-fly PySR training ────────────────────────────────────────────────

def train_pysr_chunk(chunk_idx, chunk_df, plots_dir):
    """Train PySR on a single chunk and return the best equation string.

    Returns the equation string or None on failure.
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("  ERROR: pysr not installed. Cannot train missing chunks.")
        return None

    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi
    omega = gaussian_filter1d(omega_raw, sigma=SIGMA)
    theta = np.cumsum(omega) * DT
    d_omega_dt = np.gradient(omega, DT)

    X = np.stack([theta, omega], axis=1)
    y = d_omega_dt

    # PySR temp files go into a subdirectory of the plots folder
    pysr_tmp = os.path.join(plots_dir, f"pysr_tmp_chunk_{chunk_idx:06d}")
    os.makedirs(pysr_tmp, exist_ok=True)

    model = PySRRegressor(
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
        output_directory       = pysr_tmp,
    )

    try:
        model.fit(X, y)
        hof = model.equations_
        best_row = hof.loc[hof["loss"].idxmin()]
        eq_str = str(best_row["equation"])
        loss = float(best_row["loss"])
        complexity = int(best_row["complexity"])
        print(f"  Trained PySR: eq={eq_str}  (complexity={complexity}, loss={loss:.4e})")
        return eq_str
    except Exception as e:
        print(f"  ERROR during PySR training: {e}")
        return None


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_chunk(chunk_idx, chunk_start_time, t, omega_raw, omega_smooth, omega_sim,
               rmse_sim, output_dir):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(t, omega_raw, color='#2196F3', linewidth=0.4, alpha=0.3,
            label='Empirical (raw)')
    ax.plot(t, omega_smooth, color='#4CAF50', linewidth=1.0, alpha=0.9,
            label='Smoothed ($\\sigma$=15)')
    ax.plot(t, omega_sim, color='#F44336', linewidth=1.0, alpha=0.9, linestyle='--',
            label='Forward sim')
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('$\\omega$ (rad/s)', fontsize=7)
    ax.set_title(f'Chunk {chunk_idx} — {chunk_start_time}', fontsize=7)
    ax.legend(loc='upper right', fontsize=5.5)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot PySR forward-simulated omega vs empirical omega")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help="Path to PySR results CSV (default: results_all_combined.csv)")
    parser.add_argument("--chunks", type=int, nargs="+", default=None,
                        help="Specific chunk indices to plot")
    parser.add_argument("--best", type=int, default=None,
                        help="Plot the N best chunks by rmse_omega")
    parser.add_argument("--train-missing", action="store_true",
                        help="Train PySR on-the-fly for chunks not found in CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        if args.best:
            print(f"Error: CSV not found: {args.csv}")
            return
        elif not args.train_missing:
            print(f"Error: CSV not found: {args.csv}")
            print("Use --train-missing to train PySR on-the-fly for requested chunks.")
            return
        # --train-missing with --chunks: CSV optional
        df = pd.DataFrame()
    else:
        # Load results CSV
        print(f"Loading results from {args.csv}...")
        df = pd.read_csv(args.csv)
        df["rmse_omega"] = pd.to_numeric(df["rmse_omega"], errors="coerce")

    # Determine which chunks to plot
    if args.chunks:
        chunk_indices = args.chunks
    elif args.best:
        valid = df[(df["sim_ok"] == True) & (df["rmse_omega"].notna())].sort_values("rmse_omega")
        chunk_indices = valid["chunk_id"].head(args.best).tolist()
        print(f"Plotting {args.best} best chunks by rmse_omega")
    else:
        print("Specify --chunks or --best")
        return

    # Load empirical data
    # Dataset is at dsr/dataset/ — 4 levels up from EmpiricalData/
    base = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
    parquet_path = os.path.join(base, "South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(base, "Frequency_data_SK.pkl")
    if os.path.exists(parquet_path):
        data_path = parquet_path
    elif os.path.exists(pickle_path):
        data_path = pickle_path
    else:
        print("Error: No data file found")
        return

    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    all_chunks = get_all_valid_chunks(data)
    n_chunks = len(all_chunks)
    print(f"Found {n_chunks} valid chunks")

    # Output directory next to CSV
    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    plots_dir = os.path.join(csv_dir, "plots_forward_sim")
    os.makedirs(plots_dir, exist_ok=True)

    plotted = 0
    trained = 0
    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)

        if chunk_idx >= n_chunks:
            print(f"  Chunk {chunk_idx}: out of range ({n_chunks} total), skipping")
            continue

        # Look up equation from CSV
        eq_str = None
        rmse_csv = np.nan
        if len(df) > 0:
            row = df[df["chunk_id"] == chunk_idx]
            if len(row) > 0:
                row = row.iloc[0]
                eq_str = str(row.get("equation", ""))
                rmse_csv = float(row.get("rmse_omega", np.nan))
                if not eq_str or eq_str == "nan":
                    eq_str = None

        # Train on-the-fly if missing
        if eq_str is None:
            if args.train_missing:
                chunk_start_time, chunk_df = all_chunks[chunk_idx]
                print(f"  Chunk {chunk_idx} ({chunk_start_time}): not in CSV, training PySR...")
                eq_str = train_pysr_chunk(chunk_idx, chunk_df, plots_dir)
                trained += 1
                if eq_str is None:
                    print(f"  Chunk {chunk_idx}: training failed, skipping")
                    continue
            else:
                print(f"  Chunk {chunk_idx}: not found in CSV, skipping "
                      "(use --train-missing to train on-the-fly)")
                continue

        # Load and prepare empirical data
        chunk_start_time, chunk_df = all_chunks[chunk_idx]
        t, omega_raw, omega_smooth, theta_smooth = prepare_chunk(chunk_df)

        # Parse equation and simulate ODE
        try:
            dyn_func = pysr_expr_to_func(eq_str)
            theta_sim, omega_sim = simulate_chunk(t, theta_smooth[0], omega_smooth[0], dyn_func)

            if np.any(np.isnan(omega_sim)) or np.any(np.isinf(omega_sim)):
                print(f"  Chunk {chunk_idx}: simulation diverged, skipping")
                continue
            if np.max(np.abs(omega_sim)) > 100000000000 * np.max(np.abs(omega_smooth)):
                print(f"  Chunk {chunk_idx}: simulation blew up, skipping")
                continue

            rmse_sim = np.sqrt(np.mean((omega_sim - omega_smooth) ** 2))
        except Exception as e:
            print(f"  Chunk {chunk_idx}: simulation failed ({e}), skipping")
            continue

        if np.isfinite(rmse_csv):
            print(f"  Chunk {chunk_idx} ({chunk_start_time}): "
                  f"CSV RMSE={rmse_csv:.6f}, Recomputed RMSE={rmse_sim:.6f}")
        else:
            print(f"  Chunk {chunk_idx} ({chunk_start_time}): "
                  f"RMSE={rmse_sim:.6f}  eq={eq_str}")

        plot_chunk(chunk_idx, chunk_start_time, t,
                   omega_raw, omega_smooth, omega_sim,
                   rmse_sim, plots_dir)
        plotted += 1

    print(f"\n{plotted} plots saved to: {plots_dir}")
    if trained > 0:
        print(f"({trained} chunks trained on-the-fly with PySR)")


if __name__ == "__main__":
    main()
