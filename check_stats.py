import numpy as np


def analyze_chunk():
    data = np.loadtxt("chunk_data.txt")
    # Cols: theta, omega, d_omega_dt (SCALED)
    theta = data[:, 0]
    omega = data[:, 1]
    # t = data[:, 2] # Missing in current file
    d_omega_dt = data[:, 2]

    print(f"Omega Mean: {np.mean(omega):.6f}")
    print(f"Theta Drift (Total Change): {theta[-1] - theta[0]:.6f}")
    
    
    # Check correlation
    corr_w_dot = np.corrcoef(omega, d_omega_dt)[0, 1]
    corr_theta_dot = np.corrcoef(theta, d_omega_dt)[0, 1]
    
    print(f"Correlation (Omega, dOmega/dt): {corr_w_dot:.4f}")
    print(f"Correlation (Theta, dOmega/dt): {corr_theta_dot:.4f}")

    # Check signal to noise roughly (std dev)
    print(f"Std Omega: {np.std(omega):.6f}")
    print(f"Std dOmega/dt: {np.std(d_omega_dt):.6f}")

if __name__ == "__main__":
    analyze_chunk()
