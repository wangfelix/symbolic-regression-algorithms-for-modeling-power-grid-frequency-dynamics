"""
Plot forward-simulated omega (from SINDy-recovered equation) vs empirical omega.

Reads the equation from a SINDy results CSV, re-simulates the ODE from initial
conditions, and plots against the empirical + smoothed data for visual comparison.

Usage:
    python plot_forward_sim_sindy.py --run-name run_SLURM_3753891_sindy --best 10
    python plot_forward_sim_sindy.py --run-name run_SLURM_3753891_sindy --chunks 1234 5678
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import glob
import warnings
warnings.filterwarnings("ignore")

SIGMA = 15
RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_sindy_5min_all_chunks")


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


def prepare_chunk(chunk_df, dt=1.0):
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

def parse_equation(eq_str):
    """Parse a PySINDy polynomial equation string into coefficients dict.

    PySINDy formats the constant term as 'coeff 1' (e.g. '0.052 1'), where '1'
    is the feature name for the constant basis function.
    """
    coeffs = {
        "1": 0.0,
        "theta": 0.0, "omega": 0.0,
        "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0,
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }
    eq_str = eq_str.replace("+ -", "+-").replace("- ", "-").replace("  ", " ").strip()
    terms = []
    current_term = ""
    for char in eq_str:
        if char == '+' and current_term.strip():
            terms.append(current_term.strip())
            current_term = ""
        else:
            current_term += char
    if current_term.strip():
        terms.append(current_term.strip())

    term_patterns = [
        ("theta^3", "theta^3"), ("theta^2 omega", "theta^2 omega"),
        ("theta omega^2", "theta omega^2"), ("omega^3", "omega^3"),
        ("theta^2", "theta^2"), ("theta omega", "theta omega"),
        ("omega theta", "theta omega"), ("omega^2", "omega^2"),
        ("theta", "theta"), ("omega", "omega"),
    ]
    for term in terms:
        term = term.strip()
        if not term:
            continue
        matched = False
        for pattern, coeff_key in term_patterns:
            if pattern in term:
                coeff_str = term.replace(pattern, "").replace("*", "").strip()
                try:
                    coeffs[coeff_key] = float(coeff_str) if coeff_str else 1.0
                except ValueError:
                    pass
                matched = True
                break
        if not matched:
            # Handle PySINDy constant term format: "coeff 1"
            if term.endswith(" 1"):
                try:
                    coeffs["1"] = float(term[:-2].strip())
                except ValueError:
                    pass
            else:
                try:
                    coeffs["1"] = float(term)
                except ValueError:
                    pass
    return coeffs


def omega_coeffs_from_row(row):
    """Extract full-precision omega equation coefficients from CSV numeric columns."""
    return {
        "1": float(row.get("Coeff_Const", 0) or 0),
        "theta": float(row.get("Coeff_Theta", 0) or 0),
        "omega": float(row.get("Coeff_Omega", 0) or 0),
        "theta^2": float(row.get("Coeff_Theta2", 0) or 0),
        "theta omega": float(row.get("Coeff_ThetaOmega", 0) or 0),
        "omega^2": float(row.get("Coeff_Omega2", 0) or 0),
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }


def simulate_sindy_ode(t, theta0, omega0, coeffs_theta, coeffs_omega):
    """Simulate the SINDy ODE in raw (unscaled) space using odeint."""
    # Pre-extract coefficients for performance (called at every timestep)
    tc = [coeffs_theta[k] for k in ["1", "theta", "omega", "theta^2", "theta omega", "omega^2"]]
    oc = [coeffs_omega[k] for k in ["1", "theta", "omega", "theta^2", "theta omega", "omega^2"]]

    def drift(state, t_):
        th, om = state
        dtheta = tc[0] + tc[1]*th + tc[2]*om + tc[3]*th*th + tc[4]*th*om + tc[5]*om*om
        domega = oc[0] + oc[1]*th + oc[2]*om + oc[3]*th*th + oc[4]*th*om + oc[5]*om*om
        return [dtheta, domega]

    sol = odeint(drift, np.array([theta0, omega0]), t, full_output=False)
    return sol[:, 0], sol[:, 1]  # theta_sim, omega_sim


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
        description="Plot SINDy forward-simulated omega vs empirical omega")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run folder name inside results_sindy_5min_all_chunks/")
    parser.add_argument("--chunks", type=int, nargs="+", default=None,
                        help="Specific chunk indices to plot")
    parser.add_argument("--best", type=int, default=None,
                        help="Plot the N best chunks by Sim_RMSE_Omega")
    args = parser.parse_args()

    run_dir = os.path.join(RESULTS_BASE, args.run_name)
    if not os.path.isdir(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Load results CSV
    combined_path = os.path.join(run_dir, "all_chunks_combined.csv")
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
    else:
        csv_files = sorted(glob.glob(os.path.join(run_dir, "chunks_*.csv")))
        if not csv_files:
            print(f"No CSV files found in {run_dir}")
            return
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["Chunk_Index"], keep="last")

    df["Sim_RMSE_Omega"] = pd.to_numeric(df["Sim_RMSE_Omega"], errors="coerce")

    # Determine which chunks to plot
    if args.chunks:
        chunk_indices = args.chunks
    elif args.best:
        valid = df[df["Sim_RMSE_Omega"].notna()].sort_values("Sim_RMSE_Omega")
        chunk_indices = valid["Chunk_Index"].head(args.best).tolist()
        print(f"Plotting {args.best} best chunks by Sim_RMSE_Omega")
    else:
        print("Specify --chunks or --best")
        return

    # Load empirical data
    parquet_path = os.path.join(os.path.dirname(__file__),
                                "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(os.path.dirname(__file__),
                               "../dataset/Frequency_data_SK.pkl")
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

    # Output directory
    plots_dir = os.path.join(run_dir, "plots_forward_sim")
    os.makedirs(plots_dir, exist_ok=True)

    plotted = 0
    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)

        row = df[df["Chunk_Index"] == chunk_idx]
        if len(row) == 0:
            print(f"  Chunk {chunk_idx}: not found in CSV, skipping")
            continue
        row = row.iloc[0]

        eq_omega_str = str(row.get("Eq_Omega", ""))
        eq_theta_str = str(row.get("Eq_Theta", ""))
        rmse_sim_csv = float(row.get("Sim_RMSE_Omega", np.nan))

        if not eq_omega_str or eq_omega_str == "nan" or "FAILED" in eq_omega_str:
            print(f"  Chunk {chunk_idx}: no valid omega equation, skipping")
            continue
        if not eq_theta_str or eq_theta_str == "nan" or "FAILED" in eq_theta_str:
            print(f"  Chunk {chunk_idx}: no valid theta equation, skipping")
            continue

        if chunk_idx >= n_chunks:
            print(f"  Chunk {chunk_idx}: out of range ({n_chunks} total), skipping")
            continue

        # Load and prepare empirical data
        chunk_start_time, chunk_df = all_chunks[chunk_idx]
        t, omega_raw, omega_smooth, theta_smooth = prepare_chunk(chunk_df)

        # Build ODE coefficients: full-precision numerics for omega, parse string for theta
        try:
            coeffs_theta = parse_equation(eq_theta_str)
            coeffs_omega = omega_coeffs_from_row(row)
            theta_sim, omega_sim = simulate_sindy_ode(
                t, theta_smooth[0], omega_smooth[0], coeffs_theta, coeffs_omega)

            if np.any(np.isnan(omega_sim)) or np.any(np.isinf(omega_sim)):
                print(f"  Chunk {chunk_idx}: simulation diverged, skipping")
                continue
            if np.max(np.abs(omega_sim)) > 100 * np.max(np.abs(omega_smooth)):
                print(f"  Chunk {chunk_idx}: simulation blew up, skipping")
                continue

            rmse_sim = np.sqrt(np.mean((omega_sim - omega_smooth) ** 2))
        except Exception as e:
            print(f"  Chunk {chunk_idx}: simulation failed ({e}), skipping")
            continue

        print(f"  Chunk {chunk_idx} ({chunk_start_time}): "
              f"CSV RMSE={rmse_sim_csv:.6f}, Recomputed RMSE={rmse_sim:.6f}")

        plot_chunk(chunk_idx, chunk_start_time, t,
                   omega_raw, omega_smooth, omega_sim,
                   rmse_sim, plots_dir)
        plotted += 1

    print(f"\n{plotted} plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
