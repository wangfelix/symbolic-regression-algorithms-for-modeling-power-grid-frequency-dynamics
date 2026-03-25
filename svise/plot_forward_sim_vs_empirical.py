"""
Plot forward-simulated omega (from SVISE-recovered equation) vs empirical omega.

Reads the equation from a SVISE results CSV, re-simulates the ODE from initial
conditions, and plots against the empirical data for visual comparison.

Usage:
    python plot_forward_sim_vs_empirical.py --run-dir results_5min_all_chunks/run_SLURM_3718912_combo5 --chunks 1234 5678
    python plot_forward_sim_vs_empirical.py --run-dir results_5min_all_chunks/run_SLURM_3718868_combo38 --best 5
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

T_SCALE = 30.0


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


def prepare_chunk(chunk_df, sigma=0, dt=1.0):
    """Prepare theta and omega from a chunk DataFrame."""
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi
    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega = omega_raw.copy()
    theta = np.cumsum(omega) * dt
    t = np.arange(len(omega)) * dt
    return t, theta, omega, omega_raw


def compute_scaling_params(theta, omega, t_scale=T_SCALE):
    """Recompute the same scaling params used during SVISE training (integrator model)."""
    import torch
    train_x = torch.tensor(np.stack([theta, omega], axis=1), dtype=torch.float32)
    mean_x = train_x.mean(dim=0).numpy()
    std_x = train_x.std(dim=0).numpy()
    std_x[std_x < 1e-6] = 1.0
    mean_x[1] = 0.0
    std_x[0] = std_x[1] * t_scale
    return mean_x, std_x


# ── Equation Parsing & Simulation ────────────────────────────────────────────

def parse_equation(eq_str):
    """Parse a polynomial equation string into coefficients dict."""
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
            try:
                coeffs["1"] = float(term)
            except ValueError:
                pass
    return coeffs


def simulate_ode(t, theta0, omega0, coeffs_omega, mean_x, std_x, t_scale=T_SCALE):
    """Simulate ODE in scaled space, unscale back."""
    x0 = np.array([theta0, omega0])
    x0_scaled = (x0 - mean_x) / std_x
    t_scaled = t / t_scale

    def drift(state, t_):
        th, om = state
        domega = (coeffs_omega["1"]
                  + coeffs_omega["theta"] * th + coeffs_omega["omega"] * om
                  + coeffs_omega["theta^2"] * th**2 + coeffs_omega["theta omega"] * th * om
                  + coeffs_omega["omega^2"] * om**2
                  + coeffs_omega["theta^3"] * th**3 + coeffs_omega["theta^2 omega"] * th**2 * om
                  + coeffs_omega["theta omega^2"] * th * om**2 + coeffs_omega["omega^3"] * om**3)
        return [om, domega]

    sol_scaled = odeint(drift, x0_scaled, t_scaled, full_output=False)
    sol = sol_scaled * std_x + mean_x
    return sol[:, 0], sol[:, 1]  # theta_sim, omega_sim


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_chunk(chunk_idx, chunk_start_time, t, omega_raw, omega_sim, eq_omega_str,
               eq_phys_str, rmse_gp, rmse_sim, output_dir):
    """Plot forward-simulated omega vs empirical omega for a single chunk.
    Creates two separate plots: one with raw data, one with sigma=15 smoothed data."""

    # Plot 1: raw empirical vs forward sim
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(t, omega_raw, color='#2196F3', linewidth=0.6, alpha=0.9, label='Empirical (raw)')
    ax.plot(t, omega_sim, color='#F44336', linewidth=1.0, alpha=0.9, linestyle='--',
            label='Forward sim')
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('$\\omega$ (rad/s)', fontsize=7)
    ax.set_title(f'Chunk {chunk_idx} — {chunk_start_time}', fontsize=7)
    ax.legend(loc='upper right', fontsize=5.5)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    path_raw = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_raw.png")
    plt.savefig(path_raw, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_raw}")

    # Plot 2: sigma=15 smoothed empirical vs forward sim (with raw background)
    omega_smooth15 = gaussian_filter1d(omega_raw, sigma=15)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(t, omega_raw, color='#2196F3', linewidth=0.4, alpha=0.3,
            label='Empirical (raw)')
    ax.plot(t, omega_smooth15, color='#4CAF50', linewidth=1.0, alpha=0.9,
            label='Empirical ($\\sigma$=15)')
    ax.plot(t, omega_sim, color='#F44336', linewidth=1.0, alpha=0.9, linestyle='--',
            label='Forward sim')
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('$\\omega$ (rad/s)', fontsize=7)
    ax.set_title(f'Chunk {chunk_idx} — {chunk_start_time}', fontsize=7)
    ax.legend(loc='upper right', fontsize=5.5)
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    path_smooth = os.path.join(output_dir, f"chunk_{chunk_idx:06d}_sigma15.png")
    plt.savefig(path_smooth, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_smooth}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot forward-simulated omega vs empirical")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to SVISE run directory")
    parser.add_argument("--chunks", type=int, nargs="+", default=None,
                        help="Specific chunk indices to plot")
    parser.add_argument("--best", type=int, default=None,
                        help="Plot the N best chunks by Sim_RMSE_Omega")
    parser.add_argument("--sigma", type=int, default=0,
                        help="Gaussian smoothing sigma for empirical data (default: 0 = raw)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(os.path.dirname(__file__), run_dir)

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

    for col in ["Orig_RMSE_Omega", "Sim_RMSE_Omega"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Determine which chunks to plot
    if args.chunks:
        chunk_indices = args.chunks
    elif args.best:
        valid = df[df["Sim_RMSE_Omega"].notna()].sort_values("Sim_RMSE_Omega")
        chunk_col = "Chunk_Index" if "Chunk_Index" in valid.columns else "Active_Chunk_Index"
        chunk_indices = valid[chunk_col].head(args.best).tolist()
        print(f"Plotting {args.best} best chunks by Sim_RMSE_Omega")
    else:
        print("Specify --chunks or --best")
        return

    # Load empirical data
    parquet_path = os.path.join(os.path.dirname(__file__), "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
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

    chunk_col = "Chunk_Index" if "Chunk_Index" in df.columns else "Active_Chunk_Index"

    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)

        # Get equation from CSV
        row = df[df[chunk_col] == chunk_idx]
        if len(row) == 0:
            print(f"  Chunk {chunk_idx}: not found in CSV, skipping")
            continue
        row = row.iloc[0]

        eq_omega_str = str(row.get("Eq_Omega", ""))
        eq_phys_str = str(row.get("Eq_Omega_Physical", "N/A"))
        rmse_gp = float(row.get("Orig_RMSE_Omega", np.nan))
        rmse_sim_csv = float(row.get("Sim_RMSE_Omega", np.nan))

        if not eq_omega_str or eq_omega_str == "nan" or "FAILED" in eq_omega_str:
            print(f"  Chunk {chunk_idx}: no valid equation, skipping")
            continue

        if chunk_idx >= n_chunks:
            print(f"  Chunk {chunk_idx}: out of range ({n_chunks} total), skipping")
            continue

        # Load and prepare empirical data
        chunk_start_time, chunk_df = all_chunks[chunk_idx]
        t, theta_emp, omega_emp, omega_raw = prepare_chunk(chunk_df, sigma=args.sigma)
        mean_x, std_x = compute_scaling_params(theta_emp, omega_emp)

        # Parse equation and simulate
        try:
            coeffs = parse_equation(eq_omega_str)
            theta_sim, omega_sim = simulate_ode(t, theta_emp[0], omega_emp[0], coeffs, mean_x, std_x)

            if np.any(np.isnan(omega_sim)) or np.any(np.isinf(omega_sim)):
                print(f"  Chunk {chunk_idx}: simulation diverged, skipping")
                continue
            if np.max(np.abs(omega_sim)) > 100 * np.max(np.abs(omega_emp)):
                print(f"  Chunk {chunk_idx}: simulation blew up, skipping")
                continue

            rmse_sim = np.sqrt(np.mean((omega_sim - omega_emp) ** 2))
        except Exception as e:
            print(f"  Chunk {chunk_idx}: simulation failed ({e}), skipping")
            continue

        print(f"  Chunk {chunk_idx} ({chunk_start_time}): GP RMSE={rmse_gp:.6f}, Sim RMSE={rmse_sim:.6f}")

        plot_chunk(chunk_idx, chunk_start_time, t,
                   omega_raw if args.sigma == 0 else omega_emp,
                   omega_sim, eq_omega_str, eq_phys_str,
                   rmse_gp, rmse_sim, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
