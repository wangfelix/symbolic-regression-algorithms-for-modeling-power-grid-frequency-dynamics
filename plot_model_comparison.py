import numpy as np
import matplotlib.pyplot as plt
import os

def model_9(theta, omega):
    # Model 9: 2.374108e-6 - omega^2 * theta
    return 2.374108e-6 - (omega**2 * theta)

def model_13(theta, omega):
    # Model 13: -0.74387 * omega^2 * theta
    return -0.743870008423 * (omega**2 * theta)

def simulate_model(model_func, t_eval, theta0, omega0):
    omega_sim = np.zeros_like(t_eval)
    theta_sim = np.zeros_like(t_eval)
    
    omega_sim[0] = omega0
    theta_sim[0] = theta0
    
    # Simple Euler integration
    for i in range(len(t_eval) - 1):
        curr_t = t_eval[i]
        curr_w = omega_sim[i]
        curr_th = theta_sim[i]
        dt = t_eval[i+1] - curr_t
        
        d_omega = model_func(curr_th, curr_w)
        
        omega_sim[i+1] = curr_w + d_omega * dt
        theta_sim[i+1] = curr_th + curr_w * dt
        
    return theta_sim, omega_sim

def plot_comparison():
    file_path = "chunk_data.txt"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    # Load data: theta, omega, d_omega_dt (no time column in current version)
    data = np.loadtxt(file_path)
    theta_emp = data[:, 0]
    omega_emp = data[:, 1]
    
    # Reconstruct time (assuming dt=1.0s)
    dt = 1.0
    t_emp = np.arange(len(theta_emp)) * dt
    
    # Run Models
    th9, w9 = simulate_model(model_9, t_emp, theta_emp[0], omega_emp[0])
    th13, w13 = simulate_model(model_13, t_emp, theta_emp[0], omega_emp[0])
    
    # Convert to Absolute Frequency (Assuming 60Hz base for user context)
    # The Omega is frequency deviation.
    base_freq = 60.0
    omega_emp_abs = omega_emp + base_freq
    w9_abs = w9 + base_freq
    w13_abs = w13 + base_freq
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(t_emp, omega_emp_abs, 'k-', label='Empirical Data', alpha=0.5, linewidth=2)
    plt.plot(t_emp, w9_abs, label=r'Model 9: $\dot{\omega} = 2.37e^{-6} - \omega^2 \theta$', color='green', linewidth=2)
    plt.plot(t_emp, w13_abs, label=r'Model 13: $\dot{\omega} = -0.744 \omega^2 \theta$', color='red', linestyle='--', linewidth=2)
    
    plt.title("Symbolic Regression Models vs Empirical Data (Trajectory)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend(loc='best')
    plt.grid(True)
    
    out_file = "model_comparison_plot.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    plot_comparison()
