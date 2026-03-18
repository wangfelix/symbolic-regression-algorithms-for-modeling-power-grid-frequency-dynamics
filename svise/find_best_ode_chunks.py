"""
Find chunks where the ODE simulation (using the learned equation) actually
tracks the empirical smoothed data well.

This script:
1. Reads the combined CSV to get equations for all valid chunks
2. Loads the frequency data and builds valid chunks
3. For each chunk with a valid equation, re-simulates the ODE using odeint
4. Computes "simulation RMSE" between ODE output and smoothed data
5. Ranks chunks by simulation RMSE and prints the top N

Usage:
    python find_best_ode_chunks.py
    python find_best_ode_chunks.py --top 20
    python find_best_ode_chunks.py --csv results_5min_all_chunks/all_chunks_combined.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import csv
import warnings
warnings.filterwarnings("ignore")

# Must match training
SIGMA = 10
T_SCALE = 30.0

# ── Data Loading ─────────────────────────────────────────────────────────────

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


# ── Equation Parsing & Simulation ────────────────────────────────────────────

def parse_equation(eq_str):
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
    """Simulate the ODE in scaled space, unscale back."""
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
        return [om, domega]  # dtheta/dt = omega (integrator model)

    solution_scaled = odeint(drift, x0_scaled, t_scaled, full_output=False)
    solution = solution_scaled * std_x + mean_x
    return solution[:, 0], solution[:, 1]  # theta_sim, omega_sim


def compute_scaling_params(theta, omega_smooth, t_scale=T_SCALE):
    """Recompute the same scaling params used during training (integrator model)."""
    import torch
    train_x = torch.tensor(np.stack([theta, omega_smooth], axis=1), dtype=torch.float32)
    mean_x = train_x.mean(dim=0).numpy()
    std_x = train_x.std(dim=0).numpy()
    std_x[std_x < 1e-6] = 1.0
    # Integrator constraints
    mean_x[1] = 0.0
    std_x[0] = std_x[1] * t_scale
    return mean_x, std_x


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find chunks with best ODE simulation match")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to combined CSV")
    parser.add_argument("--data", type=str,
                        default=os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl"))
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top chunks to display")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Save full results to CSV (optional)")
    args = parser.parse_args()

    # Find CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(os.path.dirname(__file__),
                                "results_5min_all_chunks", "all_chunks_combined.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return

    print(f"Reading equations from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter to valid (non-NaN) equations only
    valid_mask = df["Eq_Omega"].notna() & ~df["Eq_Omega"].str.contains("nan", case=False, na=True)
    df_valid = df[valid_mask].copy()
    print(f"Total rows: {len(df)}, Valid equations: {len(df_valid)}")

    # Load frequency data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    all_chunks = get_all_valid_chunks(data)
    n_chunks = len(all_chunks)
    print(f"Total valid data chunks: {n_chunks}")

    # Process each valid chunk
    results = []
    total = len(df_valid)

    for idx, (_, row) in enumerate(df_valid.iterrows()):
        chunk_id = int(row["Chunk_Index"])

        if chunk_id >= n_chunks:
            continue

        eq_omega_str = str(row["Eq_Omega"])

        try:
            coeffs_omega = parse_equation(eq_omega_str)
        except Exception:
            continue

        # Load and prepare data for this chunk
        _, chunk_df = all_chunks[chunk_id]
        t, theta_emp, omega_smooth, omega_raw = prepare_data(chunk_df)
        mean_x, std_x = compute_scaling_params(theta_emp, omega_smooth)

        # Simulate ODE
        try:
            theta_sim, omega_sim = simulate_ode(
                t, theta_emp[0], omega_smooth[0], coeffs_omega, mean_x, std_x)

            # Check for blow-ups
            if np.any(np.isnan(omega_sim)) or np.any(np.isinf(omega_sim)):
                continue
            if np.max(np.abs(omega_sim)) > 100 * np.max(np.abs(omega_smooth)):
                continue  # Diverged

            # Compute simulation RMSE
            sim_rmse_omega = np.sqrt(np.mean((omega_sim - omega_smooth)**2))
            sim_rmse_theta = np.sqrt(np.mean((theta_sim - theta_emp)**2))
            sim_rmse_total = np.sqrt((sim_rmse_omega**2 + sim_rmse_theta**2) / 2)

            # Also get the original (SVISE internal) RMSE for comparison
            orig_rmse_total = float(row["RMSE_Total"]) if pd.notna(row["RMSE_Total"]) else np.nan

            results.append({
                "Chunk_Index": chunk_id,
                "Chunk_Start_Time": row["Chunk_Start_Time"],
                "Sim_RMSE_Omega": sim_rmse_omega,
                "Sim_RMSE_Theta": sim_rmse_theta,
                "Sim_RMSE_Total": sim_rmse_total,
                "Orig_RMSE_Total": orig_rmse_total,
                "Final_Loss": float(row["Final_Loss"]) if pd.notna(row["Final_Loss"]) else np.nan,
                "Eq_Omega": eq_omega_str,
            })
        except Exception:
            continue

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{total} chunks... ({len(results)} non-divergent so far)")

    print(f"\nFinished. {len(results)} chunks had non-divergent ODE simulations (out of {total} valid).")

    if not results:
        print("No chunks produced valid ODE simulations.")
        return

    # Sort by simulation RMSE
    results.sort(key=lambda r: r["Sim_RMSE_Total"])

    # Print top N
    print(f"\n{'='*90}")
    print(f"  TOP {args.top} CHUNKS BY ODE SIMULATION RMSE (lower = equation tracks data better)")
    print(f"{'='*90}")
    print(f"{'Rank':>4}  {'Chunk':>6}  {'Start Time':<22}  {'Sim RMSE ω':>11}  {'Sim RMSE θ':>11}  {'Sim RMSE Tot':>12}  {'Orig RMSE':>10}")
    print(f"{'-'*90}")

    for rank, r in enumerate(results[:args.top], 1):
        print(f"{rank:>4}  {r['Chunk_Index']:>6}  {r['Chunk_Start_Time']:<22}  "
              f"{r['Sim_RMSE_Omega']:>11.6f}  {r['Sim_RMSE_Theta']:>11.6f}  "
              f"{r['Sim_RMSE_Total']:>12.6f}  {r['Orig_RMSE_Total']:>10.6f}")

    print(f"\nBest chunk for ODE simulation: Chunk {results[0]['Chunk_Index']} "
          f"({results[0]['Chunk_Start_Time']}) with Sim RMSE Total = {results[0]['Sim_RMSE_Total']:.6f}")

    # Calculate and print statistics for Sim_RMSE_Omega
    rmse_omega_all = [r["Sim_RMSE_Omega"] for r in results]
    print(f"\n{'='*90}")
    print("  STATISTICS FOR ODE SIMULATION RMSE OF OMEGA (across all valid non-divergent chunks)")
    print(f"{'='*90}")
    print(f"Mean   : {np.mean(rmse_omega_all):.6f} ({np.mean(rmse_omega_all):.2e})")
    print(f"Std Dev: {np.std(rmse_omega_all):.6f} ({np.std(rmse_omega_all):.2e})")
    print(f"Median : {np.median(rmse_omega_all):.6f} ({np.median(rmse_omega_all):.2e})")
    print(f"Min    : {np.min(rmse_omega_all):.6f} ({np.min(rmse_omega_all):.2e})")
    print(f"Max    : {np.max(rmse_omega_all):.6f} ({np.max(rmse_omega_all):.2e})")
    print(f"{'-'*90}")

    # Save full results if requested
    if args.output_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nFull results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
