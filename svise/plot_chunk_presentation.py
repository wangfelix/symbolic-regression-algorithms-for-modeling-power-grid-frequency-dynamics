"""
Plot SVISE Results for Presentation — Single Chunk Visualization

Shows empirical (raw), smoothed, and SVISE-simulated frequency data for a
given chunk. Reads the learned equation from a CSV produced by
run_analysis_5min_all_chunks.py.

Usage:
    # Plot chunk 109 (a good one from the 9:40 AM slot):
    python plot_chunk_presentation.py --chunk-id 109

    # Plot from a specific CSV:
    python plot_chunk_presentation.py --chunk-id 109 --csv results_5min_all_chunks/all_chunks_combined.csv

    # Save to a specific folder:
    python plot_chunk_presentation.py --chunk-id 109 --output-dir plots_presentation/
"""
import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

# Best hyperparameters (must match training)
SIGMA = 10
T_SCALE = 30.0

# =============================================================================
# Data Loading
# =============================================================================

def load_data(data_path, limit_interpolation=10):
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
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))
    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == 300:
            valid_chunks.append((chunk_start, group))
    return valid_chunks


def prepare_data(chunk_df, sigma=SIGMA, dt=1.0):
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    if sigma > 0:
        omega_smooth = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega_smooth = omega_raw.copy()

    theta = np.cumsum(omega_smooth) * dt
    t = np.arange(len(omega_smooth)) * dt
    return t, theta, omega_smooth, omega_raw


# =============================================================================
# Equation Parsing & Simulation (from plot_results_5min_chunk.py)
# =============================================================================

def parse_equation(eq_str):
    """Parse polynomial equation string into coefficients dict."""
    coeffs = {
        "1": 0.0,
        "theta": 0.0, "omega": 0.0,
        "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0,
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }

    eq_str = eq_str.replace(" + -", " + -").replace("+ -", "+-").replace("- ", "-")
    eq_str = eq_str.replace("  ", " ").strip()

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
        ("theta^3", "theta^3"),
        ("theta^2 omega", "theta^2 omega"),
        ("theta omega^2", "theta omega^2"),
        ("omega^3", "omega^3"),
        ("theta^2", "theta^2"),
        ("theta omega", "theta omega"),
        ("omega theta", "theta omega"),
        ("omega^2", "omega^2"),
        ("theta", "theta"),
        ("omega", "omega"),
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


def create_drift_function(coeffs_omega, coeffs_theta=None):
    def drift(state, t):
        theta, omega = state

        if coeffs_theta:
            dtheta_dt = sum(
                coeffs_theta.get(k, 0.0) * v for k, v in [
                    ("1", 1.0), ("theta", theta), ("omega", omega),
                    ("theta^2", theta**2), ("theta omega", theta*omega), ("omega^2", omega**2),
                    ("theta^3", theta**3), ("theta^2 omega", theta**2*omega),
                    ("theta omega^2", theta*omega**2), ("omega^3", omega**3),
                ]
            )
        else:
            dtheta_dt = omega

        domega_dt = sum(
            coeffs_omega.get(k, 0.0) * v for k, v in [
                ("1", 1.0), ("theta", theta), ("omega", omega),
                ("theta^2", theta**2), ("theta omega", theta*omega), ("omega^2", omega**2),
                ("theta^3", theta**3), ("theta^2 omega", theta**2*omega),
                ("theta omega^2", theta*omega**2), ("omega^3", omega**3),
            ]
        )

        return [dtheta_dt, domega_dt]
    return drift


def simulate_model_scaled(t, theta0, omega0, coeffs_omega, mean_x, std_x,
                          t_scale=T_SCALE, coeffs_theta=None):
    """Simulate in scaled space and unscale back."""
    x0 = np.array([theta0, omega0])
    x0_scaled = (x0 - mean_x) / std_x

    t_scaled = t / t_scale

    drift = create_drift_function(coeffs_omega, coeffs_theta)
    solution_scaled = odeint(drift, x0_scaled, t_scaled)

    solution = solution_scaled * std_x + mean_x
    return solution[:, 0], solution[:, 1]


def compute_scaling_params(theta, omega_smooth, model_type="integrator", t_scale=T_SCALE):
    """Recompute the same scaling params used during training."""
    import torch
    train_x = torch.tensor(np.stack([theta, omega_smooth], axis=1), dtype=torch.float32)
    mean_x = train_x.mean(dim=0).numpy()
    std_x = train_x.std(dim=0).numpy()
    std_x[std_x < 1e-6] = 1.0

    if model_type == "integrator":
        mean_x[1] = 0.0
        std_x[0] = std_x[1] * t_scale

    return mean_x, std_x


# =============================================================================
# Plotting
# =============================================================================

def plot_omega(t, omega_raw, omega_smooth, omega_sim, chunk_start, chunk_id,
               eq_omega_str, rmse_omega, output_dir):
    """Create a presentation-quality omega (frequency deviation) plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, omega_raw, color="#AAAAAA", alpha=0.5, linewidth=0.8,
            label="Raw $\\omega$ (empirical)")
    ax.plot(t, omega_smooth, color="#2196F3", linewidth=2.0, alpha=0.9,
            label=f"Smoothed $\\omega$ ($\\sigma={SIGMA}$)")
    ax.plot(t, omega_sim, color="#E91E63", linewidth=2.0, alpha=0.9,
            linestyle="--", label="SVISE simulation")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Deviation $\\omega$ (rad/s)")
    ax.set_title(f"SVISE Model vs. Empirical Data — Chunk {chunk_id} ({chunk_start})")

    ax.legend(loc="upper right", framealpha=0.9)

    # Add RMSE and equation text box
    textstr = f"RMSE $\\omega$: {rmse_omega:.6f}\n$d\\omega/dt$ = {_format_eq_short(eq_omega_str)}"
    props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', alpha=0.85, edgecolor='#CCCCCC')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace', bbox=props)

    plt.tight_layout()
    path = os.path.join(output_dir, f"omega_chunk_{chunk_id}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_theta(t, theta_emp, theta_sim, chunk_start, chunk_id,
               eq_theta_str, rmse_theta, output_dir):
    """Create a presentation-quality theta (phase) plot."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(t, theta_emp, color="#2196F3", linewidth=2.0, alpha=0.9,
            label="Empirical $\\theta$ (smoothed)")
    ax.plot(t, theta_sim, color="#E91E63", linewidth=2.0, alpha=0.9,
            linestyle="--", label="SVISE simulation")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase $\\theta$ (rad)")
    ax.set_title(f"SVISE Phase Estimate — Chunk {chunk_id} ({chunk_start})")

    ax.legend(loc="upper right", framealpha=0.9)

    textstr = f"RMSE $\\theta$: {rmse_theta:.6f}\n$d\\theta/dt$ = {_format_eq_short(eq_theta_str)}"
    props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', alpha=0.85, edgecolor='#CCCCCC')
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace', bbox=props)

    plt.tight_layout()
    path = os.path.join(output_dir, f"theta_chunk_{chunk_id}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def plot_combined(t, omega_raw, omega_smooth, omega_sim, theta_emp, theta_sim,
                  chunk_start, chunk_id, eq_omega_str, eq_theta_str,
                  rmse_omega, rmse_theta, output_dir):
    """Create a combined 2-panel plot (theta + omega)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Theta panel
    axes[0].plot(t, theta_emp, color="#2196F3", linewidth=2.0, alpha=0.9,
                 label="Empirical $\\theta$")
    axes[0].plot(t, theta_sim, color="#E91E63", linewidth=2.0, alpha=0.9,
                 linestyle="--", label="SVISE simulation")
    axes[0].set_ylabel("Phase $\\theta$ (rad)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].set_title(f"SVISE Model vs. Empirical Data — Chunk {chunk_id} ({chunk_start})")

    # Omega panel
    axes[1].plot(t, omega_raw, color="#AAAAAA", alpha=0.5, linewidth=0.8,
                 label="Raw $\\omega$")
    axes[1].plot(t, omega_smooth, color="#2196F3", linewidth=2.0, alpha=0.9,
                 label=f"Smoothed $\\omega$ ($\\sigma={SIGMA}$)")
    axes[1].plot(t, omega_sim, color="#E91E63", linewidth=2.0, alpha=0.9,
                 linestyle="--", label="SVISE simulation")
    axes[1].set_ylabel("Freq. Deviation $\\omega$ (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Metrics text
    textstr = (f"RMSE $\\omega$: {rmse_omega:.6f}  |  RMSE $\\theta$: {rmse_theta:.6f}\n"
               f"$d\\omega/dt$ = {_format_eq_short(eq_omega_str)}")
    props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', alpha=0.85, edgecolor='#CCCCCC')
    axes[1].text(0.02, 0.02, textstr, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='bottom', fontfamily='monospace', bbox=props)

    plt.tight_layout()
    path = os.path.join(output_dir, f"combined_chunk_{chunk_id}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def _format_eq_short(eq_str, max_len=80):
    """Truncate equation for display if too long."""
    if len(eq_str) > max_len:
        return eq_str[:max_len] + "..."
    return eq_str


# =============================================================================
# Main
# =============================================================================

def find_best_csv(results_dir):
    """Find the combined CSV or the largest individual CSV."""
    combined = os.path.join(results_dir, "all_chunks_combined.csv")
    if os.path.exists(combined):
        return combined

    csvs = sorted(glob.glob(os.path.join(results_dir, "chunks_*.csv")), key=os.path.getsize, reverse=True)
    if csvs:
        return csvs[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot SVISE results for a single chunk (presentation quality)")
    parser.add_argument("--chunk-id", type=int, required=True,
                        help="Chunk index to plot (from the CSV Chunk_Index column)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with chunk results. If not given, "
                             "searches results_5min_all_chunks/ for all_chunks_combined.csv")
    parser.add_argument("--data", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl"),
                        help="Path to the frequency data pickle file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: plots_presentation/)")
    args = parser.parse_args()

    # Find CSV
    if args.csv:
        csv_path = args.csv
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks")
        csv_path = find_best_csv(results_dir)
        if csv_path is None:
            print("Error: No CSV file found. Run aggregate_results_5min_all_chunks.py first, "
                  "or specify --csv manually.")
            return

    print(f"Reading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Find chunk
    row = df[df["Chunk_Index"] == args.chunk_id]
    if len(row) == 0:
        print(f"Error: Chunk {args.chunk_id} not found in CSV.")
        print(f"Available chunk range: {df['Chunk_Index'].min()} - {df['Chunk_Index'].max()}")
        return

    row = row.iloc[-1]  # Take last if duplicates
    eq_theta_str = str(row["Eq_Theta"])
    eq_omega_str = str(row["Eq_Omega"])
    rmse_omega = float(row["RMSE_Omega"]) if pd.notna(row["RMSE_Omega"]) else float('nan')
    rmse_theta = float(row["RMSE_Theta"]) if pd.notna(row["RMSE_Theta"]) else float('nan')

    if "nan" in eq_omega_str.lower():
        print(f"Warning: Chunk {args.chunk_id} has NaN equations (training failed). "
              "Only empirical data will be plotted.")

    print(f"Chunk {args.chunk_id}: {row['Chunk_Start_Time']}")
    print(f"  Eq_Theta: {eq_theta_str}")
    print(f"  Eq_Omega: {eq_omega_str}")
    print(f"  RMSE_Omega: {rmse_omega:.6f}" if not np.isnan(rmse_omega) else "  RMSE_Omega: NaN")

    # Load data
    print(f"\nLoading data from {args.data}...")
    data = load_data(args.data)
    all_chunks = get_all_valid_chunks(data)

    if args.chunk_id >= len(all_chunks):
        print(f"Error: Chunk {args.chunk_id} out of range (dataset has {len(all_chunks)} valid chunks).")
        return

    chunk_start, chunk_df = all_chunks[args.chunk_id]
    t, theta_emp, omega_smooth, omega_raw = prepare_data(chunk_df)

    # Output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "plots_presentation")
    os.makedirs(output_dir, exist_ok=True)

    # Simulate model (if equations are valid)
    has_valid_eq = "nan" not in eq_omega_str.lower()

    if has_valid_eq:
        coeffs_omega = parse_equation(eq_omega_str)

        # For integrator model: theta eq is always "omega" (enforced by the model)
        coeffs_theta = None
        if "omega" not in eq_theta_str.lower() or eq_theta_str.strip() == "1.00000omega":
            coeffs_theta = None  # Integrator: dtheta/dt = omega
        else:
            coeffs_theta = parse_equation(eq_theta_str)

        # Compute scaling params (same as training)
        mean_x, std_x = compute_scaling_params(theta_emp, omega_smooth)

        theta_sim, omega_sim = simulate_model_scaled(
            t, theta_emp[0], omega_smooth[0], coeffs_omega,
            mean_x, std_x, t_scale=T_SCALE, coeffs_theta=coeffs_theta)

        print(f"\nSimulation completed. Generating plots...")
    else:
        # Plot without simulation
        theta_sim = np.full_like(t, np.nan)
        omega_sim = np.full_like(t, np.nan)
        print(f"\nTraining failed for this chunk. Plotting empirical data only...")

    # Generate plots
    plot_omega(t, omega_raw, omega_smooth, omega_sim, chunk_start, args.chunk_id,
               eq_omega_str, rmse_omega, output_dir)

    plot_theta(t, theta_emp, theta_sim, chunk_start, args.chunk_id,
               eq_theta_str, rmse_theta, output_dir)

    plot_combined(t, omega_raw, omega_smooth, omega_sim, theta_emp, theta_sim,
                  chunk_start, args.chunk_id, eq_omega_str, eq_theta_str,
                  rmse_omega, rmse_theta, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
