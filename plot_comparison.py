import numpy as np
import matplotlib.pyplot as plt
import os

def model_7(theta, omega, t):
    # Model 7: 15.632... 
    # d_omega/dt = 6.136491e-05 + (omega / ((theta/omega) + 1))
    
    denom = (theta/omega) + 1 if abs(omega) > 1e-9 else 1.0
    if abs(denom) < 1e-9:
        term = 0.0
    else:
        term = omega / denom
        
    return 6.136491e-05 + term

def model_9(theta, omega, t):
    # Model 9: 15.31...
    try:
        if theta > 10: 
            term_A = theta / np.pi
        else:
            e_theta = np.exp(theta)
            if e_theta < 0: return 0
            sqrt_val = np.sqrt(e_theta)
            if sqrt_val > 700: denom_part = float('inf')
            else: denom_part = np.exp(sqrt_val)
                
            if denom_part == 0: term_A = 0 
            else:
                sub = t / denom_part
                denom_A = np.pi - sub
                if abs(denom_A) < 1e-9: term_A = 0
                else: term_A = theta / denom_A
                        
        val = -0.000110812521 + term_A**2
        return val
    except Exception:
        return 0.0

def model_10(theta, omega, t):
    # Model 10: 15.20...
    try:
        const_C = 45.172 # Precalculated
        
        if theta > 10: 
            term_A = theta / np.pi
        else:
            e_theta = np.exp(theta)
            sqrt_val = np.sqrt(e_theta)
            if sqrt_val > 700: denom_part = float('inf')
            else: denom_part = np.exp(sqrt_val)
            
            sub = t / denom_part
            denom_A = np.pi - sub
            if abs(denom_A) < 1e-9: term_A = 0
            else: term_A = theta / denom_A
        
        val = -0.000219292314 + (term_A / const_C)
        return val
    except Exception:
        return 0.0

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
        
        d_omega = model_func(curr_th, curr_w, curr_t)
        
        omega_sim[i+1] = curr_w + d_omega * dt
        theta_sim[i+1] = curr_th + curr_w * dt
        
    return theta_sim, omega_sim

def plot_comparison():
    file_path = "chunk_data.txt"
    if not os.path.exists(file_path):
        print("Data file not found.")
        return

    data = np.loadtxt(file_path)
    theta_emp = data[:, 0]
    omega_emp = data[:, 1]
    t_emp = data[:, 2]
    
    # Run Models
    th7, w7 = simulate_model(model_7, t_emp, theta_emp[0], omega_emp[0])
    th9, w9 = simulate_model(model_9, t_emp, theta_emp[0], omega_emp[0])
    th10, w10 = simulate_model(model_10, t_emp, theta_emp[0], omega_emp[0])
    
    # Convert to Absolute Frequency (Assuming 60Hz base)
    base_freq = 60.0
    omega_emp_abs = omega_emp + base_freq
    w7_abs = w7 + base_freq
    w9_abs = w9 + base_freq
    w10_abs = w10 + base_freq
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(t_emp, omega_emp_abs, 'k-', label='Empirical', alpha=0.3, linewidth=2)
    plt.plot(t_emp, w7_abs, label='Model 7', color='blue', alpha=0.8)
    plt.plot(t_emp, w9_abs, label='Model 9', color='green', alpha=0.8)
    plt.plot(t_emp, w10_abs, label='Model 10', color='red', linestyle='--', alpha=0.8)
    
    plt.title("Symbolic Regression Models vs Empirical Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim(59.97, 60.02)
    plt.legend()
    plt.grid(True)
    
    out_file = "all_models_plot.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    plot_comparison()
