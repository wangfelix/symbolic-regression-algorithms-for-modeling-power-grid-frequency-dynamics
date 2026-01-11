import numpy as np
from scipy.integrate import odeint

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate():
    print("Generating Lorenz data...")
    # Time points
    t = np.linspace(0, 50, 5000) # 5000 points, delta t = 0.01
    
    # Check if we should use finer grain? 5000 points for 50s is 100Hz equivalent.
    
    # Initial state
    state0 = [1.0, 1.0, 1.0]
    
    # Integrate
    states = odeint(lorenz, state0, t)
    
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    # Calculate exact derivatives for validation/target
    # We want to recover dz/dt = xy - beta*z
    # So inputs should be x, y, z. Output dzdt.
    
    sigma=10; rho=28; beta=8/3
    
    dzdt = x * y - beta * z
    
    # Prepare data for AI Feynman
    # Format: x1, x2, ..., y
    # Let's see if we can give it x, y, z as inputs to predict dzdt
    
    data = np.column_stack((x, y, z, dzdt))
    
    filename = "lorenz_z_data.txt"
    np.savetxt(filename, data)
    print(f"Data saved to {filename}. Columns: x, y, z, dz/dt")

if __name__ == "__main__":
    generate()
