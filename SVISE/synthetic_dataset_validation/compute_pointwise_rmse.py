"""
Calculate point-wise mean and median RMSE (Absolute Error) between new synthetic datasets and empirical dataset.

Usage:
    python compute_pointwise_rmse.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import data_cleaning


def load_real_data(data_path):
    """Load and clean real frequency data, return omega (rad/s)."""
    print(f"Loading real data from {data_path}...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=10)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
        freq = data.loc[data['QI'] == 0, 'freq'].dropna()
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=10)
        freq = data['freq'].dropna()

    freq_clean = data_cleaning(freq, freq_limits=(59, 61))
    freq_clean = freq_clean.dropna()

    omega = (freq_clean.values - 60.0) * 2 * np.pi
    print(f"  Real data: {len(omega)} samples, mean={np.mean(omega):.6f}, std={np.std(omega):.6f}")
    return omega


def load_synthetic_data(data_path):
    """Load synthetic omega data (handles .npy generated directly from the DataFrame)."""
    print(f"Loading synthetic data from {data_path}...")
    if data_path.endswith('.npy'):
        omega = np.load(data_path)
    else:
        df = pd.read_pickle(data_path)
        omega = df['omega'].values
    print(f"  Synthetic data: {len(omega)} samples, mean={np.mean(omega):.6f}, std={np.std(omega):.6f}")
    return omega


def compute_pointwise_error(sim, real):
    """
    Computes standard RMSE and point-wise MAE between two sequences.
    Returns:
      - Mean RMSE (Root Mean Squared Error)
      - Median RMSE (Root Median Squared Error) 
      - Mean Absolute Error (MAE)
    """
    n_compare = min(len(sim), len(real))
    err = np.abs(sim[:n_compare] - real[:n_compare])
    
    mean_rmse = float(np.sqrt(np.mean(err**2)))
    median_rmse = float(np.sqrt(np.median(err**2)))
    mae = float(np.mean(err))
    
    return mean_rmse, median_rmse, mae


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Resolve real data path
    parquet_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    pickle_path = os.path.join(script_dir, "../../dataset/South_Korea_2024-08-15_2025-08-31_1s.pkl")
    
    if os.path.exists(parquet_path):
        real_path = parquet_path
    elif os.path.exists(pickle_path):
        real_path = pickle_path
    else:
        print(f"Error: Real data not found. Tried:\n  {parquet_path}\n  {pickle_path}")
        sys.exit(1)

    # Synthetic data paths (use the converted .npy ones to avoid segfaults)
    synth_without_wiener_path = os.path.join(script_dir, "synthetic_without_wiener.npy")
    synth_with_wiener_path = os.path.join(script_dir, "synthetic_with_wiener.npy")
    
    for path in [synth_without_wiener_path, synth_with_wiener_path]:
        if not os.path.exists(path):
            print(f"Error: Synthetic data not found at {path}")
            sys.exit(1)

    # Load data
    omega_real = load_real_data(real_path)
    omega_synth_noiseless = load_synthetic_data(synth_without_wiener_path)
    omega_synth_noisy = load_synthetic_data(synth_with_wiener_path)

    # Prepare smoothed empirical data
    print("\nApplying Gaussian smoothing (sigma=15) to real data...")
    omega_real_smoothed = gaussian_filter1d(omega_real, sigma=15)
    
    # Prepare smoothed noisy synthetic data
    print("Applying Gaussian smoothing (sigma=15) to noisy synthetic data...")
    omega_synth_noisy_smoothed = gaussian_filter1d(omega_synth_noisy, sigma=15)

    print("\n" + "="*50)
    print("COMPUTING RMSE & POINTWISE ERRORS")
    print("="*50)

    # 1. Noiseless vs Real (raw)
    rmse1, med_rmse1, mae1 = compute_pointwise_error(omega_synth_noiseless, omega_real)
    print(f"Dataset A1 (Noiseless vs Raw Empirical):")
    print(f"  Mean RMSE                      : {rmse1:.8f}")
    print(f"  Median RMSE                    : {med_rmse1:.8f}")
    print(f"  Mean Absolute Error (MAE)      : {mae1:.8f}\n")

    # 2. Noiseless vs Real (smoothed)
    rmse2, med_rmse2, mae2 = compute_pointwise_error(omega_synth_noiseless, omega_real_smoothed)
    print(f"Dataset A2 (Noiseless vs Smoothed Empirical):")
    print(f"  Mean RMSE                      : {rmse2:.8f}")
    print(f"  Median RMSE                    : {med_rmse2:.8f}")
    print(f"  Mean Absolute Error (MAE)      : {mae2:.8f}\n")

    # 3. Noisy vs Real (raw)
    rmse3, med_rmse3, mae3 = compute_pointwise_error(omega_synth_noisy, omega_real)
    print(f"Dataset B1 (Noisy vs Raw Empirical):")
    print(f"  Mean RMSE                      : {rmse3:.8f}")
    print(f"  Median RMSE                    : {med_rmse3:.8f}")
    print(f"  Mean Absolute Error (MAE)      : {mae3:.8f}\n")

    # 4. Noisy (smoothed) vs Real (smoothed)
    rmse4, med_rmse4, mae4 = compute_pointwise_error(omega_synth_noisy_smoothed, omega_real_smoothed)
    print(f"Dataset B2 (Noisy Smoothed vs Smoothed Empirical):")
    print(f"  Mean RMSE                      : {rmse4:.8f}")
    print(f"  Median RMSE                    : {med_rmse4:.8f}")
    print(f"  Mean Absolute Error (MAE)      : {mae4:.8f}\n")

    print("="*50)

    # Save results
    results = {
        "description": "Mean RMSE, Median RMSE, and Mean Absolute Error between new synthetic datasets and empirical data.",
        "comparisons": {
            "Dataset_A1_Noiseless_vs_RawReal": {
                "mean_rmse": rmse1,
                "median_rmse": med_rmse1,
                "mean_absolute_error_mae": mae1
            },
            "Dataset_A2_Noiseless_vs_SmoothedReal": {
                "mean_rmse": rmse2,
                "median_rmse": med_rmse2,
                "mean_absolute_error_mae": mae2
            },
            "Dataset_B1_Noisy_vs_RawReal": {
                "mean_rmse": rmse3,
                "median_rmse": med_rmse3,
                "mean_absolute_error_mae": mae3
            },
            "Dataset_B2_NoisySmoothed_vs_SmoothedReal": {
                "mean_rmse": rmse4,
                "median_rmse": med_rmse4,
                "mean_absolute_error_mae": mae4
            }
        }
    }

    results_path = os.path.join(script_dir, "rmse_pointwise_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
