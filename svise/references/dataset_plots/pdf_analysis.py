import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def create_pdf_plot(df, plots_dir):
    """
    Create PDF (Probability Density Function) analysis plots
    
    Parameters:
    df: DataFrame with frequency data
    plots_dir: Directory to save plots
    """
    print("Creating PDF plot...")
    
    # Get frequency data
    freq_data = df['freq'].dropna()
    
    plt.figure(figsize=(12, 8))

    # Plot 1: Histogram and PDF
    plt.subplot(2, 2, 1)
    plt.hist(freq_data, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Frequency Distribution (Histogram)', fontsize=12, fontweight='bold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Plot 2: Log-scale histogram
    plt.subplot(2, 2, 2)
    plt.hist(freq_data, bins=100, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.yscale('log')
    plt.title('Frequency Distribution (Log Scale)', fontsize=12, fontweight='bold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density (log scale)')
    plt.grid(True, alpha=0.3)

    # Plot 3: Q-Q plot for normality
    plt.subplot(2, 2, 3)
    stats.probplot(freq_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 4: Box plot
    plt.subplot(2, 2, 4)
    plt.boxplot(freq_data, vert=True)
    plt.title('Box Plot', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'frequency_pdf_analysis.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    print(f"PDF plot saved to: {os.path.join(plots_dir, 'frequency_pdf_analysis.png')}")

    # Print statistics
    print(f"\nFrequency statistics:")
    print(f"Mean: {freq_data.mean():.6f} Hz")
    print(f"Std: {freq_data.std():.6f} Hz")
    print(f"Skewness: {stats.skew(freq_data):.6f}")
    print(f"Kurtosis (Pearson): {stats.kurtosis(freq_data, fisher=False):.6f}")
    print(f"Min: {freq_data.min():.6f} Hz")
    print(f"Max: {freq_data.max():.6f} Hz")
    print(f"Range: {freq_data.max() - freq_data.min():.6f} Hz")
    
    return freq_data

if __name__ == "__main__":
    # Load data
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'df_South_Korea_cleansed_2024-08-15_2024-12-10.pkl')
    with open(dataset_path, 'rb') as f:
        df = pd.read_pickle(f)
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Run PDF analysis
    freq_data = create_pdf_plot(df, plots_dir)
