import os
import sys
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Ensure svise is in path if not installed as package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path}...")
    data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:,'freq'] = data.loc[:,'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(),'QI'] = 2
        data.loc[~data['freq'].isna(),'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)

    return data

def get_valid_chunk(data):
    print("Filtering for valid 15-minute chunks...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()

    # Group by 15min
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('15T'))
    
    # Filter for full chunks (900 samples at 1Hz)
    valid_chunks = [group for _, group in chunk_groups if len(group) == 900]
    
    if not valid_chunks:
        raise ValueError("No valid 15-minute chunks found in the dataset.")
        
    print(f"Found {len(valid_chunks)} valid chunks. Using the first one.")
    return valid_chunks[0]

def prepare_data(chunk_df, dt=1.0, sigma=60):
    freq_values = chunk_df['freq'].values
    
    if np.mean(freq_values) > 55:
        omega_raw = freq_values - 60.0
    else:
        omega_raw = freq_values
        
    omega = gaussian_filter1d(omega_raw, sigma=sigma)
    theta = np.cumsum(omega) * dt
    X = np.stack([theta, omega], axis=1)
    
    return X, omega_raw

def check_data():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    data = load_data(DATA_PATH)
    chunk_df = get_valid_chunk(data)
    
    sigma = 60.0
    dt = 1.0
    
    print(f"\nChecking First Chunk (Sigma={sigma})...")
    X, omega_raw = prepare_data(chunk_df, dt=dt, sigma=sigma)
    theta = X[:, 0]
    omega = X[:, 1]
    
    print(f"Data Shape: {X.shape}")
    print(f"Theta Range: [{np.min(theta):.4f}, {np.max(theta):.4f}]")
    print(f"Omega Range: [{np.min(omega):.4f}, {np.max(omega):.4f}]")
    print(f"Omega Raw Range: [{np.min(omega_raw):.4f}, {np.max(omega_raw):.4f}]")
    
    # Check for NaNs/Infs
    if np.isnan(X).any():
        print("!!! DETECTED NANs IN PROCESSED DATA !!!")
    else:
        print("No NaNs in processed data.")
        
    if np.isinf(X).any():
        print("!!! DETECTED INFs IN PROCESSED DATA !!!")
    else:
        print("No Infs in processed data.")
        
    # Check smoothness / derivatives
    d_omega = np.diff(omega) / dt
    print(f"Max abs(d_omega/dt): {np.max(np.abs(d_omega)):.6f}")
    
    # Check Scaling
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    print(f"Means: {mean_x}")
    print(f"Stds: {std_x}")
    
    if np.any(std_x < 1e-6):
         print("!!! WARNING: Very low standard deviation detected. Scaling might be unstable. !!!")

if __name__ == "__main__":
    check_data()
