import numpy as np
import matplotlib.pyplot as plt

def model_function(theta, omega, t):
    """
    The discovered AI Feynman model:
    sqrt(-666 * [ (t^2/theta + omega) - ((t^2/theta + omega) + sqrt(pi)) + sqrt(pi) ])
    note: x0=theta, x1=omega, x2=t
    """
    pi = np.pi
    
    # Calculate the inner term A
    # Avoid division by zero if theta is 0
    if abs(theta) < 1e-9:
        term_A = 0 + omega
    else:
        term_A = (t**2 / theta) + omega
        
    # The full expression inside the bracket
    # bracket = term_A - (term_A + sqrt(pi)) + sqrt(pi)
    bracket = term_A - (term_A + np.sqrt(pi)) + np.sqrt(pi)
    
    # This should be 0 (or incredibly close to it due to float precision)
    # The outer term is sqrt(-666 * bracket)
    # If bracket is slightly negative due to float error, sqrt will be nan.
    # We'll take abs to be safe for visualization, but mathematically it's 0.
    
    val_inside = -666.0 * bracket
    
    # Handle float precision noise near 0
    if abs(val_inside) < 1e-9:
        return 0.0
        
    if val_inside < 0:
        return 0.0 # Can't take sqrt of negative
        
    return np.sqrt(val_inside)

def simulate_and_plot():
    # Load empirical data
    # Columns: theta, omega, t, d_omega_dt
    data = np.loadtxt("chunk_data.txt")
    theta_empirical = data[:, 0]
    omega_empirical = data[:, 1]
    t = data[:, 2]
    
    # Initial conditions
    omega_sim = [omega_empirical[0]]
    theta_sim = [theta_empirical[0]]
    dt = t[1] - t[0] # Assuming 1.0s
    
    print(f"Starting simulation. dt={dt}, N={len(t)}")
    
    # Euler Integration
    for i in range(len(t) - 1):
        curr_t = t[i]
        curr_omega = omega_sim[-1]
        curr_theta = theta_sim[-1]
        
        # Calculate d_omega/dt from model
        d_omega_dt = model_function(curr_theta, curr_omega, curr_t)
        
        # Update Omega
        new_omega = curr_omega + d_omega_dt * dt
        
        # Update Theta (integral of omega deviation)
        # Omega is Absolute (around 60), so deviation is (Omega - 60)
        # But wait, did we define Theta as integral of deviation in previous step?
        # Yes, in regression.py: theta = cumsum(omega_dev) * dt
        new_theta = curr_theta + (curr_omega - 60.0) * dt
        
        omega_sim.append(new_omega)
        theta_sim.append(new_theta)
        
    omega_sim = np.array(omega_sim)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, omega_empirical, label='Empirical Data (Absolute)', color='blue', alpha=0.6)
    plt.plot(t, omega_sim, label='Simulated Model', color='red', linestyle='--', linewidth=2)
    
    plt.title("Model Verification: Empirical vs Simulated Frequency")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.legend()
    plt.grid(True)
    
    output_file = "model_simulation_plot.png"
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    simulate_and_plot()
