"""
Plot SVISE Results

This script generates comparison plots between empirical data and the learned SDE model.
It reads from results/<timestamp>/model.json and saves plots to results/<timestamp>/plots/.

Usage:
    python plot_results.py --timestamp 20260111_155428
    python plot_results.py --all  # Plot all available results
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint

# Ensure svise is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_data(data_path, limit_interpolation=10):
    """Load and preprocess the frequency data."""
    data = pd.read_pickle(data_path)
    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)
    return data


def get_valid_chunk(data):
    """Get the first valid 15-minute chunk."""
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()
    
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('15min'))
    valid_chunks = [group for _, group in chunk_groups if len(group) == 900]
    
    if not valid_chunks:
        raise ValueError("No valid 15-minute chunks found.")
    
    return valid_chunks[0]


def prepare_data(chunk_df, sigma=5, dt=1.0):
    """Prepare theta and omega from frequency data."""
    freq_values = chunk_df['freq'].values
    
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi
    
    if sigma > 0:
        omega = gaussian_filter1d(omega_raw, sigma=sigma)
    else:
        omega = omega_raw
    
    theta = np.cumsum(omega) * dt
    t = np.arange(len(omega)) * dt
    
    return t, theta, omega, omega_raw

def simulate_model_scaled(t, theta0, omega0, coeffs_omega, scaling_params, coeffs_theta=None):
    """
    Simulate the model in the SCALED domain and transform back.
    
    Model was learned on:
      X_scaled = (X - mean) / std
      t_scaled = t / t_scale
      dX_scaled/dt_scaled = f(X_scaled)
    """
    mean = np.array(scaling_params["mean_x"])
    std = np.array(scaling_params["std_x"])
    t_scale = scaling_params["t_scale"]
    
    # Scale initial conditions
    # X = [theta, omega]
    x0 = np.array([theta0, omega0])
    x0_scaled = (x0 - mean) / std
    
    # Scale time
    t_scaled = t / t_scale
    
    drift = create_drift_function(coeffs_omega, coeffs_theta)
    
    # Simulate in scaled domain
    # odeint expects drift(y, t)
    # The drift function returns [dtheta_scaled/dt_scaled, domega_scaled/dt_scaled]
    solution_scaled = odeint(drift, x0_scaled, t_scaled)
    
    # Unscale solution
    # X = X_scaled * std + mean
    solution = solution_scaled * std + mean
    
    return solution[:, 0], solution[:, 1]


def parse_equation(eq_str):
    """
    Parse a polynomial equation string into coefficients.
    Returns a dictionary mapping term names to coefficients.
    Example: "-0.15341theta + -0.00003omega" -> {"theta": -0.15341, "omega": -0.00003}
    """
    coeffs = {"1": 0.0, "theta": 0.0, "omega": 0.0, "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0}
    
    # Normalize the string: remove extra spaces, fix "+ -" patterns
    eq_str = eq_str.replace(" + -", " + -").replace("+ -", "+-").replace("- ", "-")
    eq_str = eq_str.replace("  ", " ").strip()
    
    # Split by + but keep - attached to numbers
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
    
    for term in terms:
        term = term.strip()
        if not term:
            continue
        
        # Determine which variable this term contains
        if "theta^2" in term:
            coeff_str = term.replace("theta^2", "").replace("*", "").strip()
            coeffs["theta^2"] = float(coeff_str) if coeff_str else 1.0
        elif "omega^2" in term:
            coeff_str = term.replace("omega^2", "").replace("*", "").strip()
            coeffs["omega^2"] = float(coeff_str) if coeff_str else 1.0
        elif "theta omega" in term or "omega theta" in term:
            coeff_str = term.replace("theta omega", "").replace("omega theta", "").replace("*", "").strip()
            coeffs["theta omega"] = float(coeff_str) if coeff_str else 1.0
        elif "theta" in term:
            coeff_str = term.replace("theta", "").replace("*", "").strip()
            coeffs["theta"] = float(coeff_str) if coeff_str else 1.0
        elif "omega" in term:
            coeff_str = term.replace("omega", "").replace("*", "").strip()
            coeffs["omega"] = float(coeff_str) if coeff_str else 1.0
        else:
            # Constant term
            try:
                coeffs["1"] = float(term)
            except ValueError:
                pass
    
    return coeffs


def create_drift_function(coeffs_omega, coeffs_theta=None):
    """Create a drift function from parsed coefficients."""
    def drift(state, t):
        theta, omega = state
        
        # d(theta)/dt
        if coeffs_theta:
             dtheta_dt = (
                coeffs_theta["1"] +
                coeffs_theta["theta"] * theta +
                coeffs_theta["omega"] * omega +
                coeffs_theta["theta^2"] * theta**2 +
                coeffs_theta["theta omega"] * theta * omega +
                coeffs_theta["omega^2"] * omega**2
            )
        else:
            # d(theta)/dt = omega (always enforced in IntegratorSDE or fallback)
            dtheta_dt = omega
        
        # d(omega)/dt from learned polynomial
        domega_dt = (
            coeffs_omega["1"] +
            coeffs_omega["theta"] * theta +
            coeffs_omega["omega"] * omega +
            coeffs_omega["theta^2"] * theta**2 +
            coeffs_omega["theta omega"] * theta * omega +
            coeffs_omega["omega^2"] * omega**2
        )
        
        return [dtheta_dt, domega_dt]
    
    return drift


def simulate_model(t, theta0, omega0, coeffs_omega):
    """Simulate the learned SDE model (deterministic part only)."""
    drift = create_drift_function(coeffs_omega)
    initial_state = [theta0, omega0]
    solution = odeint(drift, initial_state, t)
    return solution[:, 0], solution[:, 1]  # theta_sim, omega_sim


def format_equation_latex(eq_str, var_name="\\omega"):
    """Format equation string for matplotlib legend."""
    # Simple cleanup for display
    eq = eq_str.replace("theta", "\\theta").replace("omega", "\\omega")
    eq = eq.replace("^2", "^2").replace(" + -", " - ")
    return f"$\\frac{{d{var_name}}}{{dt}} = {eq}$"


def plot_results(timestamp, results_dir, data_path):
    """Generate plots for a specific experiment."""
    result_folder = os.path.join(results_dir, timestamp)
    json_path = os.path.join(result_folder, "model.json")
    
    if not os.path.exists(json_path):
        print(f"No model.json found at {json_path}")
        return
    
    # Load model config
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    config = model_data["config"]
    equations = model_data["equations"]
    
    print(f"Plotting results for timestamp: {timestamp}")
    print(f"Model: {config.get('model', 'unknown')}, Sigma: {config.get('sigma', 5)}")
    
    # Load empirical data
    data = load_data(data_path)
    chunk_df = get_valid_chunk(data)
    
    sigma = config.get("sigma", 5)
    t, theta_emp, omega_filtered, omega_raw = prepare_data(chunk_df, sigma=sigma)
    
    # Parse equations
    theta_eq_str = equations[0] if len(equations) > 0 else "N/A"
    omega_eq_str = equations[1] if len(equations) > 1 else equations[0]
    
    coeffs_omega = parse_equation(omega_eq_str)
    
    coeffs_theta = None
    if config.get('model') == 'sparse' and len(equations) > 1:
         theta_eq_str = equations[0]
         coeffs_theta = parse_equation(theta_eq_str)
         print(f"Parsed theta coefficients: {coeffs_theta}")
    
    print(f"Parsed omega coefficients: {coeffs_omega}")
    
    # Simulate model
    if "scaling_params" in model_data:
        print("Using scaled simulation...")
        scaling_params = model_data["scaling_params"]

        theta_sim, omega_sim = simulate_model_scaled(t, theta_emp[0], omega_filtered[0], coeffs_omega, scaling_params, coeffs_theta=coeffs_theta)
    else:
        print("WARNING: No scaling parameters found. Simulating with raw coefficients (likely incorrect).")
        theta_sim, omega_sim = simulate_model(t, theta_emp[0], omega_filtered[0], coeffs_omega)
    
    # Create plots directory
    plots_dir = os.path.join(result_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Theta
    theta_eq_latex = format_equation_latex(theta_eq_str, var_name="\\theta")
    axes[0].plot(t, theta_emp, 'b-', label='Empirical $\\theta$', alpha=0.7, linewidth=1.5)
    axes[0].plot(t, theta_sim, 'r--', label=f'Model: {theta_eq_latex}', alpha=0.9, linewidth=1.5)
    axes[0].set_ylabel('Phase $\\theta$ (rad)', fontsize=12)
    # Move legend outside or make it smaller if equation is long
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=9, fancybox=True, shadow=True)
    axes[0].set_title(f'SVISE Model vs Empirical Data (Timestamp: {timestamp})', fontsize=14, y=1.2)
    
    # Plot Omega
    omega_eq_latex = format_equation_latex(omega_eq_str, var_name="\\omega")
    axes[1].plot(t, omega_raw, 'b-', label='Empirical $\\omega$ (Raw)', alpha=0.3, linewidth=1.0)
    axes[1].plot(t, omega_filtered, 'g-', label=f'Filtered $\\omega$ ($\sigma={sigma}$)', alpha=0.8, linewidth=1.5)
    axes[1].plot(t, omega_sim, 'r--', label=f'Model: {omega_eq_latex}', alpha=0.9, linewidth=1.5)
    axes[1].set_ylabel('Frequency Deviation $\\omega$ (rad/s)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=9, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save main comparison plot
    plot_path = os.path.join(plots_dir, "comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()
    
    # Create equation summary plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    eq_text = f"Model: {config.get('model', 'unknown')}\n"
    eq_text += f"Sigma: {config.get('sigma', 5)}, Degree: {config.get('degree', 2)}\n"
    eq_text += f"Final Loss: {model_data.get('final_loss', 'N/A'):.2f}\n\n"
    eq_text += "Learned Equations:\n"
    eq_text += f"  dθ/dt = {equations[0]}\n" if len(equations) > 0 else ""
    eq_text += f"  dω/dt = {equations[1]}\n" if len(equations) > 1 else ""
    
    ax.text(0.1, 0.9, eq_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    summary_path = os.path.join(plots_dir, "equation_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {summary_path}")
    plt.close()
    
    print(f"Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Plot SVISE results")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Timestamp of the result to plot (e.g., 20260111_155428)")
    parser.add_argument("--all", action="store_true",
                        help="Plot all available results")
    parser.add_argument("--data", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl"),
                        help="Path to the frequency data file")
    
    args = parser.parse_args()
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    
    if args.all:
        # Plot all results
        timestamps = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d))]
        for ts in sorted(timestamps):
            plot_results(ts, results_dir, args.data)
    elif args.timestamp:
        plot_results(args.timestamp, results_dir, args.data)
    else:
        # Default: plot the most recent result
        timestamps = [d for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d))]
        if timestamps:
            latest = sorted(timestamps)[-1]
            print(f"No timestamp specified. Using latest: {latest}")
            plot_results(latest, results_dir, args.data)
        else:
            print("No results found. Run run_analysis.py first.")


if __name__ == "__main__":
    main()
