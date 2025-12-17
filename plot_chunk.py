import numpy as np
import matplotlib.pyplot as plt
import os

def plot_chunk_data():
    file_path = "chunk_data.txt"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load data: columns are [theta, omega, t, d_omega_dt]
    data = np.loadtxt(file_path)
    theta = data[:, 0]
    omega = data[:, 1]
    t = data[:, 2]
    d_omega_dt = data[:, 3]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Omega
    axes[0].plot(t, omega, label='Omega (Freq Deviation)', color='blue')
    axes[0].set_ylabel('Omega [Hz]')
    axes[0].set_title('Frequency Deviation (Omega)')
    axes[0].grid(True)
    axes[0].legend()

    # Plot dOmega/dt
    axes[1].plot(t, d_omega_dt, label='dOmega/dt', color='red', alpha=0.7)
    axes[1].set_ylabel('dOmega/dt [Hz/s]')
    axes[1].set_title('Rate of Change (dOmega/dt)')
    axes[1].grid(True)
    axes[1].legend()

    # Plot Theta
    axes[2].plot(t, theta, label='Theta (Phase)', color='green')
    axes[2].set_ylabel('Theta [rad]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Phase (Theta)')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    output_file = "chunk_data_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_chunk_data()
