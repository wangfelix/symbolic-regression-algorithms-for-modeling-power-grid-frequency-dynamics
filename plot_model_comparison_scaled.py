import numpy as np
import matplotlib.pyplot as plt
import os

# Physical Constants (for legend) / Scaling Factors
SCALE_OMEGA = 100.0
SCALE_DOMEGA = 10000.0

def model_1(theta, omega, x1):
    # Model 1: 0.0216 * pi
    # y = 0.0678...
    # Constant drift
    return 0.021613219918 * np.pi

def model_2(theta, omega, x1):
    # Model 2: -0.00204 * x1^-1
    if abs(x1) < 1e-9: return 0.0
    return -0.002047073408 * (1.0 / x1)

def model_4(theta, omega, x1):
    # Model 4: sqrt(0.0939 * x1)
    validation_val = 0.093936620489 * x1
    # Handle domain error for sqrt
    if validation_val < 0: return 0.0 # Or complex?
    return np.sqrt(validation_val)

def model_5(theta, omega, x1):
    # Model 5: sqrt( 7.38e-6 * (x0 - (x1 + x0/x1)) )
    # x0 = theta, x1 = omega_scaled
    
    if abs(x1) < 1e-9: return 0.0
    
    term = theta - (x1 + (theta/x1))
    val = 0.000007381195 * term
    
    if val < 0: return 0.0
    return np.sqrt(val)

def simulate_model(model_func, t_eval, theta0, omega0):
    omega_sim = np.zeros_like(t_eval)
    theta_sim = np.zeros_like(t_eval)
    
    omega_sim[0] = omega0
    theta_sim[0] = theta0
    
    for i in range(len(t_eval) - 1):
        curr_t = t_eval[i]
        curr_w = omega_sim[i]
        curr_th = theta_sim[i]
        dt = t_eval[i+1] - curr_t
        
        # Calculate scaled input x1
        x1 = curr_w * SCALE_OMEGA
        
        # Calculate scaled output y
        y_scaled = model_func(curr_th, curr_w, x1)
        
        # Unscale to get physical d_omega/dt
        d_omega = y_scaled / SCALE_DOMEGA
        
        omega_sim[i+1] = curr_w + d_omega * dt
        theta_sim[i+1] = curr_th + curr_w * dt
        
    return theta_sim, omega_sim

def plot_comparison():
    file_path = "chunk_data.txt"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    # Load data
    # IMPORTANT: The file now contains SCALED columns: theta, omega*100, d_omega*10000
    data = np.loadtxt(file_path)
    theta_emp = data[:, 0]
    omega_scaled_emp = data[:, 1]
    
    # Recover physical omega
    omega_emp = omega_scaled_emp / SCALE_OMEGA
    
    dt = 1.0
    t_emp = np.arange(len(theta_emp)) * dt
    
    # Run Models
    th1, w1 = simulate_model(model_1, t_emp, theta_emp[0], omega_emp[0])
    th2, w2 = simulate_model(model_2, t_emp, theta_emp[0], omega_emp[0])
    th4, w4 = simulate_model(model_4, t_emp, theta_emp[0], omega_emp[0])
    th5, w5 = simulate_model(model_5, t_emp, theta_emp[0], omega_emp[0])
    
    # Convert to Absolute Frequency
    base_freq = 60.0
    omega_emp_abs = omega_emp + base_freq
    w1_abs = w1 + base_freq
    w2_abs = w2 + base_freq
    w4_abs = w4 + base_freq
    w5_abs = w5 + base_freq
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(t_emp, omega_emp_abs, 'k-', label='Empirical', alpha=0.3, linewidth=2)
    plt.plot(t_emp, w1_abs, label='Model 1 (Const)', color='blue', alpha=0.8)
    plt.plot(t_emp, w2_abs, label='Model 2 (Inv)', color='orange', alpha=0.8)
    plt.plot(t_emp, w4_abs, label='Model 4 (Sqrt)', color='green', alpha=0.8)
    plt.plot(t_emp, w5_abs, label='Model 5 (Complex)', color='red', linestyle='--', alpha=0.8)
    
    plt.title("Scaled Models vs Empirical Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True)
    
    out_file = "scaled_models_plot.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    plot_comparison()
