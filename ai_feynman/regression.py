
import pandas as pd
import numpy as np
import aifeynman
import os
from scipy.ndimage import gaussian_filter1d

def load_data(data_path, limit_interploation=10):
    '''
    Function to load and preprocess data.
    Parameters:
    ----
    data_path (str): Path to the data file.
    limit_interploation (int): Limit for interpolation of missing values. Default is 10.
    Returns:
    ----
    pd.DataFrame: Preprocessed DataFrame with missing values interpolated and quality indicators updated.
    '''
    
    data = pd.read_pickle(data_path)

    ''' Add missing bad quality indicator'''
    # Ensure QI column exists or handle it safely if not? 
    # User code assumes 'QI' exists.
    
    if 'QI' in data.columns:
        data_0 = data.loc[data['QI']==0]
        if not data_0.empty and data_0['freq'].isna().any():
             # This line from user code might crush if index is empty, added check
             ind_candidates = data_0[data_0['freq'].isna()].index
             if not ind_candidates.empty:
                 ind = ind_candidates[0]
                 data.loc[ind,'QI'] = 2

    # Interpolation
    data.loc[:,'freq'] = data.loc[:,'freq'].interpolate(method='time', limit=limit_interploation)
    
    if 'QI' in data.columns:
        data.loc[data['freq'].isna(),'QI'] = 2
        data.loc[~data['freq'].isna(),'QI'] = 0
        
    return data

def load_data_full_hours(data):
    '''
    Function to load dataframe and filter it to keep only full hours.
    Parameters:
    data (pd.DataFrame): DataFrame to be filtered.
    Returns: 
    pd.DataFrame: Filtered DataFrame with only full hours.
    '''
    # Step 1: Load the data
    # Assuming standard index is datetime
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna()
    else:
        data_filtered = data[data['freq'].notna()].dropna()

    # Step 2: Group the data by the hour
    hourly_groups = data_filtered.groupby(data_filtered.index.floor('h'))

    # Step 3: Filter out incomplete hours 
    # South Korea data is usually 1Hz (1 sample per sec), so 3600 samples per hour.
    valid_hours = hourly_groups.filter(lambda x: len(x) == 3600)
    
    return valid_hours

def prepare_chunk_for_aifeynman(chunk_df, dt=1.0, sigma=60):
    """
    Prepare a single DataFrame chunk for AI Feynman.
    Calculates omega, theta, d_omega_dt.
    
    Returns:
        np.array: Data matrix [theta, omega, t, d_omega_dt] or similar
                  columns suitable for finding d_omega_dt = f(theta, omega, t)
    """
    # 1. Omega (freq deviation)
    # Assuming 'freq' is already the deviation or raw frequency. 
    # If raw (around 60Hz), we should subtract mean or nominal.
    # The user mentioned "frequency deviation in mHz" in the other code, but "data['freq']" here.
    # Usually grid frequency is centered around 60Hz (Korea).
    
    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55: # It's likely raw frequency (60Hz)
        omega_raw = freq_values - 60.0
    else:
        omega_raw = freq_values # It's likely already deviation
    
    # 2. Gaussian Filter
    # Apply filtering to get the smooth drift component
    print(f"Applying Gaussian filter with sigma={sigma}...")
    omega = gaussian_filter1d(omega_raw, sigma=sigma)
        
    # 3. Theta (phase) -> Integral of omega
    # theta(t) = theta(0) + integral(omega) dt
    # Using the filtered omega for consistent phase-space trajectory
    theta = np.cumsum(omega) * dt
    
    # 4. d_omega_dt -> Derivative of omega
    # Using the filtered omega to get the 'drift' derivative
    d_omega_dt = np.gradient(omega, dt)
    
    # 5. Time t
    # Relative time from start of chunk
    t_numeric = np.arange(len(omega)) * dt
    
    # Prepare data for AI Feynman
    # Target: d_omega_dt
    # Features: t, theta, omega
    # AI Feynman expects format: x1, x2, ..., xn, y
    # We want to find: y = f(x1, x2, ...)
    # y = d_omega_dt
    # args: t, theta, omega
    
    # Let's order them: theta, omega, t, d_omega_dt (Target last)
    data_matrix = np.column_stack([theta, omega, t_numeric, d_omega_dt])
    
    return data_matrix

def run_analysis():
    data_path = "./dataset/Frequency_data_SK.pkl"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        # Try finding it relative to current script
        data_path = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
        
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    
    print("Filtering full hours...")
    # This returns one big DF with only valid hours, but we want to split by hour
    # Actually load_data_full_hours returns the filtered DF. We can group it again.
    valid_data_df = load_data_full_hours(data)
    
    # Group by hour to get chunks
    grouped = valid_data_df.groupby(valid_data_df.index.floor('h'))
    
    # Pick the first chunk for demonstration
    chunk_timestamp, chunk_df = next(iter(grouped))
    print(f"Processing chunk: {chunk_timestamp}")
    print(f"Chunk shape: {chunk_df.shape}")
    
    # Prepare data
    # Assuming 1Hz data -> dt=1.0s
    # Applying sigma=60 as suggested by user
    data_matrix = prepare_chunk_for_aifeynman(chunk_df, dt=1.0, sigma=60)
    
    filename = "chunk_data.txt"
    np.savetxt(filename, data_matrix)
    print(f"Data saved to {filename}. Columns: theta, omega, t, d_omega_dt")
    
    # Run AI Feynman
    # aifeynman.run_aifeynman(pathdir, filename, BF_try_time, BF_ops_file_type, polyfit_deg=3, NN_epochs=500)
    # pathdir: directory containing .dat file (usually './')
    # filename: name of file
    # BF_try_time: time limit in seconds
    # BF_ops_file_type: file with allowed operations (e.g., '14ops.txt')
    
    print("Starting AI Feynman run...")
    # We need to ensure 14ops.txt or similar exists, or pass specific args
    # If 14ops.txt is not present, we might need to create it or point to it.
    # aifeynman usually comes with it, but let's check.
    
    try:
        aifeynman.run_aifeynman('./', filename, 60, "14ops.txt", polyfit_deg=3, NN_epochs=4000)
    except Exception as e:
        print(f"AI Feynman execution failed: {e}")

if __name__ == "__main__":
    run_analysis()
