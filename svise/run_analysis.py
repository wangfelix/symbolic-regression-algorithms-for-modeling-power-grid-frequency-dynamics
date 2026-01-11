import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
import argparse
import csv
import json
import datetime
import shutil

# Ensure svise is in path if not installed as package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import svise
from svise.sde_learning import SparsePolynomialSDE, SparsePolynomialIntegratorSDE

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path}...")
    data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:,'freq'] = data.loc[:,'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(),'QI'] = 2
        data.loc[~data['freq'].isna(),'QI'] = 0
    else:
        # Interpolate anyway just in case
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
    
    # Calculate Frequency Deviation
    if np.mean(freq_values) > 55:
        omega_raw = freq_values - 60.0
    else:
        omega_raw = freq_values
        
    # Gaussian Filter
    omega = gaussian_filter1d(omega_raw, sigma=sigma)
    
    # Theta (Integral of omega)
    theta = np.cumsum(omega) * dt
    
    # Time vector
    t = np.arange(len(omega)) * dt
    
    # Combine into State Matrix X = [theta, omega]
    X = np.stack([theta, omega], axis=1)
    
    return t, X, omega_raw

def save_experiment_results(args, loss, equations, start_time):
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Save detailed JSON
    result_folder = os.path.join(log_dir, timestamp)
    os.makedirs(result_folder, exist_ok=True)
    
    result_data = {
        "config": vars(args),
        "timestamp": timestamp,
        "final_loss": loss,
        "equations": equations
    }
    
    json_path = os.path.join(result_folder, "model.json")
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"Detailed results saved to {json_path}")
    
    # 2. Append to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "experiments_log.csv")
    file_exists = os.path.isfile(csv_path)
    
    # Parse equations into columns (Eq_0, Eq_1)
    eq_theta = equations[0] if len(equations) > 0 else "N/A"
    eq_omega = equations[1] if len(equations) > 1 else "N/A"
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Sigma", "Degree", "LR", "Epochs", "FinalLoss", "Equation_Theta", "Equation_Omega"])
        
        writer.writerow([
            timestamp,
            args.model,
            args.sigma,
            args.degree,
            args.lr,
            args.epochs,
            f"{loss:.4f}" if isinstance(loss, float) else str(loss),
            eq_theta,
            eq_omega
        ])
    print(f"Summary appended to {csv_path}")

def run_experiment(args):
    start_time = datetime.datetime.now()
    
    # 1. Config
    DATA_PATH = os.path.join(os.path.dirname(__file__), "../dataset/Frequency_data_SK.pkl")
    DT = 1.0
    
    # 2. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    data = load_data(DATA_PATH)
    chunk_df = get_valid_chunk(data)
    
    # 3. Prepare Data
    t_np, X_np, freq_raw = prepare_data(chunk_df, dt=DT, sigma=args.sigma)
    
    # Convert to Torch Tensors
    train_t = torch.tensor(t_np, dtype=torch.float32)
    train_x = torch.tensor(X_np, dtype=torch.float32) # Observations: [theta, omega]
    
    # 4. Global Scaling
    mean_x = train_x.mean(dim=0)
    std_x = train_x.std(dim=0)
    std_x[std_x < 1e-6] = 1.0 # Avoid division by zero
    
    train_x_scaled = (train_x - mean_x) / std_x
    print(f"Data Scaled. Means: {mean_x}, Stds: {std_x}")

    # 5. Model Setup
    d = 2 
    num_meas = 2 
    G = torch.eye(d) 
    measurement_noise = torch.tensor([1e-4, 1e-4]) 
    
    # Scale time
    t_scale = 90.0
    train_t_scaled = train_t / t_scale
    t_span = (train_t_scaled[0], train_t_scaled[-1])

    print(f"Initializing {args.model}...")
    
    common_params = {
        "d": d,
        "t_span": t_span,
        "degree": args.degree,
        "n_reparam_samples": 30,
        "G": G,
        "num_meas": num_meas,
        "measurement_noise": measurement_noise,
        "tau": 1e-5,
        "train_t": train_t_scaled,
        "train_x": train_x_scaled,
        "input_labels": ["theta", "omega"],
    }
    
    if args.model == "sparse":
        model = SparsePolynomialSDE(
            **common_params,
            n_tau=50,
            n_quad=256,
            quad_percent=0.5
        )
    elif args.model == "integrator":
         model = SparsePolynomialIntegratorSDE(
             **common_params,
             n_tau=50,
         )
         pass
    else:
        raise ValueError(f"Unknown model type: {args.model}")
        
    # 6. Training Loop
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    
    print(f"Starting training for {epochs} epochs...")
    final_loss = float('nan')
    loss_history = {}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ELBO computation
        loss = -model.elbo(train_t_scaled, train_x_scaled, beta=1.0, N=len(train_t))
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
            loss_history[epoch] = loss.item()
        
        final_loss = loss.item()
            
    # 7. Results
    print("\nTraining Complete.")
    print("Learned Drift Parameters (Equation terms):")
    
    equation_strings = []
    try:
        equation_strings = model.sde_prior.get_feature_names()
        for i, eq in enumerate(equation_strings):
            print(f"dx_{i}/dt = {eq}")
    except Exception as e:
        print(f"Could not extract equations directly: {e}")
        print(model)
        equation_strings = ["Error extracting"] * d

    # update json structure in save_experiment_results to include loss_history, I need to pass it
    # modifying save_experiment_results signature required or updating it inside the function
    # Let me redefine save_experiment_results above or inside run_experiment?
    # Actually I should have edited save_experiment_results first or included it here.
    # Since I'm replacing a huge chunk, I'll include save_experiment_results update here.
    
    save_experiment_results(args, final_loss, equation_strings, start_time, loss_history)

def save_experiment_results(args, loss, equations, start_time, loss_history=None):
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Save detailed JSON
    result_folder = os.path.join(log_dir, timestamp)
    os.makedirs(result_folder, exist_ok=True)
    
    result_data = {
        "config": vars(args),
        "timestamp": timestamp,
        "final_loss": loss,
        "equations": equations,
        "loss_history": loss_history
    }
    
    json_path = os.path.join(result_folder, "model.json")
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"Detailed results saved to {json_path}")
    
    # 2. Append to CSV (Already handled in previous block of this function)
    # Re-pasting the CSV part to ensure consistency
    csv_path = os.path.join(os.path.dirname(__file__), "experiments_log.csv")
    file_exists = os.path.isfile(csv_path)
    
    # Parse equations into columns (Eq_0, Eq_1)
    eq_theta = equations[0] if len(equations) > 0 else "N/A"
    eq_omega = equations[1] if len(equations) > 1 else "N/A"
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Sigma", "Degree", "LR", "Epochs", "FinalLoss", "Equation_Theta", "Equation_Omega"])
        
        writer.writerow([
            timestamp,
            args.model,
            args.sigma,
            args.degree,
            args.lr,
            args.epochs,
            f"{loss:.4f}" if isinstance(loss, float) else str(loss),
            eq_theta,
            eq_omega
        ])
    print(f"Summary appended to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run SVISE Analysis on Power Grid Data")
    
    parser.add_argument("--model", type=str, default="sparse", choices=["sparse", "integrator"],
                        help="Model type: 'sparse' (SparsePolynomialSDE) or 'integrator' (SparsePolynomialIntegratorSDE)")
    parser.add_argument("--sigma", type=float, default=5,
                        help="Sigma for Gaussian filtering of frequency data")
    parser.add_argument("--degree", type=int, default=2,
                        help="Maximum degree of polynomial drift function")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
                        
    args = parser.parse_args()
    
    print("Configuration:")
    print(json.dumps(vars(args), indent=2))
    
    run_experiment(args)
    
if __name__ == "__main__":
    main()
