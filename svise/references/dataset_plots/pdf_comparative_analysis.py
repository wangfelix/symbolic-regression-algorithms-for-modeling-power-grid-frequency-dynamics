import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def create_comparative_pdf_plots(df, plots_dir):
    """
    Create comparative PDF plots for different time periods
    
    Parameters:
    df: DataFrame with frequency data
    plots_dir: Directory to save plots
    """
    print("Creating comparative PDF plots...")
    
    # Create PDF subdirectory
    pdf_dir = os.path.join(plots_dir, 'pdf')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    # Get frequency data
    freq_data = df['freq'].dropna()
    
    # 1. Summer vs Winter PDF
    print("Creating Summer vs Winter PDF comparison...")
    create_seasonal_pdf(df, pdf_dir)
    
    # 2. Day vs Night PDF
    print("Creating Day vs Night PDF comparison...")
    create_day_night_pdf(df, pdf_dir)
    
    # 3. Weekday vs Weekend PDF
    print("Creating Weekday vs Weekend PDF comparison...")
    create_weekday_weekend_pdf(df, pdf_dir)

def create_seasonal_pdf(df, pdf_dir):
    """Create PDF comparison for Summer vs Winter"""
    
    # Define seasons based on months
    # Summer: June, July, August (6, 7, 8)
    # Winter: December, January, February (12, 1, 2)
    summer_months = [6, 7, 8]
    winter_months = [12, 1, 2]
    
    # Get data for each season
    summer_data = df[df.index.month.isin(summer_months)]['freq'].dropna()
    winter_data = df[df.index.month.isin(winter_months)]['freq'].dropna()
    
    print(f"Summer data: {len(summer_data):,} points")
    print(f"Winter data: {len(winter_data):,} points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot both PDFs
    plt.hist(summer_data, bins=100, density=True, alpha=0.7, color='orange', 
             edgecolor='black', label=f'Summer (n={len(summer_data):,})')
    plt.hist(winter_data, bins=100, density=True, alpha=0.7, color='lightblue', 
             edgecolor='black', label=f'Winter (n={len(winter_data):,})')
    
    plt.title('Frequency Distribution: Summer vs Winter', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xlim(59.90, 60.10)  # Set consistent x-axis limits
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Summer:
μ={summer_data.mean():.4f} Hz, σ={summer_data.std():.4f} Hz
s={stats.skew(summer_data):.4f}, κ={stats.kurtosis(summer_data, fisher=False):.4f}
Winter:
μ={winter_data.mean():.4f} Hz, σ={winter_data.std():.4f} Hz
s={stats.skew(winter_data):.4f}, κ={stats.kurtosis(winter_data, fisher=False):.4f}"""
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_summer_vs_winter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summer vs Winter PDF saved to: {os.path.join(pdf_dir, 'frequency_pdf_summer_vs_winter.png')}")
    
    # Create log-scale version
    plt.figure(figsize=(12, 8))
    plt.hist(winter_data, bins=100, density=True, alpha=0.7, color='lightblue', 
             edgecolor='black', label=f'Winter (n={len(winter_data):,})')
    plt.hist(summer_data, bins=100, density=True, alpha=0.7, color='orange', 
             edgecolor='black', label=f'Summer (n={len(summer_data):,})')
    plt.yscale('log')
    plt.title('Frequency Distribution: Summer vs Winter (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_summer_vs_winter_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summer vs Winter PDF (log scale) saved to: {os.path.join(pdf_dir, 'frequency_pdf_summer_vs_winter_log.png')}")

def create_day_night_pdf(df, pdf_dir):
    """Create PDF comparison for Day vs Night"""
    
    # Define day (6 AM - 6 PM) and night (6 PM - 6 AM)
    day_hours = list(range(6, 18))  # 6 AM to 5 PM
    night_hours = list(range(18, 24)) + list(range(0, 6))  # 6 PM to 5 AM
    
    # Get data for each period
    day_data = df[df.index.hour.isin(day_hours)]['freq'].dropna()
    night_data = df[df.index.hour.isin(night_hours)]['freq'].dropna()
    
    print(f"Day data: {len(day_data):,} points")
    print(f"Night data: {len(night_data):,} points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot both PDFs
    plt.hist(day_data, bins=100, density=True, alpha=0.7, color='gold', 
             edgecolor='black', label=f'Day 6AM-6PM (n={len(day_data):,})')
    plt.hist(night_data, bins=100, density=True, alpha=0.7, color='navy', 
             edgecolor='black', label=f'Night 6PM-6AM (n={len(night_data):,})')
    
    plt.title('Frequency Distribution: Day vs Night', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xlim(59.90, 60.10)  # Set consistent x-axis limits
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Day:
μ={day_data.mean():.4f} Hz, σ={day_data.std():.4f} Hz
s={stats.skew(day_data):.4f}, κ={stats.kurtosis(day_data, fisher=False):.4f}
Night:
μ={night_data.mean():.4f} Hz, σ={night_data.std():.4f} Hz
s={stats.skew(night_data):.4f}, κ={stats.kurtosis(night_data, fisher=False):.4f}"""
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_day_vs_night.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Day vs Night PDF saved to: {os.path.join(pdf_dir, 'frequency_pdf_day_vs_night.png')}")
    
    # Create log-scale version
    plt.figure(figsize=(12, 8))
    plt.hist(night_data, bins=100, density=True, alpha=0.7, color='navy', 
             edgecolor='black', label=f'Night 6PM-6AM (n={len(night_data):,})')
    plt.hist(day_data, bins=100, density=True, alpha=0.7, color='gold', 
             edgecolor='black', label=f'Day 6AM-6PM (n={len(day_data):,})')
    plt.yscale('log')
    plt.title('Frequency Distribution: Day vs Night (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_day_vs_night_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Day vs Night PDF (log scale) saved to: {os.path.join(pdf_dir, 'frequency_pdf_day_vs_night_log.png')}")

def create_weekday_weekend_pdf(df, pdf_dir):
    """Create PDF comparison for Weekday vs Weekend"""
    
    # Define weekday (Monday=0 to Friday=4) and weekend (Saturday=5, Sunday=6)
    weekday_data = df[df.index.weekday < 5]['freq'].dropna()  # Monday to Friday
    weekend_data = df[df.index.weekday >= 5]['freq'].dropna()  # Saturday and Sunday
    
    print(f"Weekday data: {len(weekday_data):,} points")
    print(f"Weekend data: {len(weekend_data):,} points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot both PDFs
    plt.hist(weekday_data, bins=100, density=True, alpha=0.7, color='green', 
             edgecolor='black', label=f'Weekday Mon-Fri (n={len(weekday_data):,})')
    plt.hist(weekend_data, bins=100, density=True, alpha=0.7, color='purple', 
             edgecolor='black', label=f'Weekend Sat-Sun (n={len(weekend_data):,})')
    
    plt.title('Frequency Distribution: Weekday vs Weekend', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xlim(59.90, 60.10)  # Set consistent x-axis limits
    plt.legend(fontsize=18, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Weekday:
μ={weekday_data.mean():.4f} Hz, σ={weekday_data.std():.4f} Hz
s={stats.skew(weekday_data):.4f}, κ={stats.kurtosis(weekday_data, fisher=False):.4f}
Weekend:
μ={weekend_data.mean():.4f} Hz, σ={weekend_data.std():.4f} Hz
s={stats.skew(weekend_data):.4f}, κ={stats.kurtosis(weekend_data, fisher=False):.4f}"""
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_weekday_vs_weekend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weekday vs Weekend PDF saved to: {os.path.join(pdf_dir, 'frequency_pdf_weekday_vs_weekend.png')}")
    
    # Create log-scale version
    plt.figure(figsize=(12, 8))
    plt.hist(weekday_data, bins=100, density=True, alpha=0.7, color='green', 
             edgecolor='black', label=f'Weekday Mon-Fri (n={len(weekday_data):,})')
    plt.hist(weekend_data, bins=100, density=True, alpha=0.7, color='purple', 
             edgecolor='black', label=f'Weekend Sat-Sun (n={len(weekend_data):,})')
    plt.yscale('log')
    plt.title('Frequency Distribution: Weekday vs Weekend (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_weekday_vs_weekend_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Weekday vs Weekend PDF (log scale) saved to: {os.path.join(pdf_dir, 'frequency_pdf_weekday_vs_weekend_log.png')}")

def create_detailed_seasonal_analysis(df, pdf_dir):
    """Create detailed seasonal analysis with all 4 seasons"""
    
    print("Creating detailed 4-season analysis...")
    
    # Define all 4 seasons
    spring_months = [3, 4, 5]  # March, April, May
    summer_months = [6, 7, 8]  # June, July, August
    autumn_months = [9, 10, 11]  # September, October, November
    winter_months = [12, 1, 2]  # December, January, February
    
    # Get data for each season
    spring_data = df[df.index.month.isin(spring_months)]['freq'].dropna()
    summer_data = df[df.index.month.isin(summer_months)]['freq'].dropna()
    autumn_data = df[df.index.month.isin(autumn_months)]['freq'].dropna()
    winter_data = df[df.index.month.isin(winter_months)]['freq'].dropna()
    
    print(f"Spring data: {len(spring_data):,} points")
    print(f"Summer data: {len(summer_data):,} points")
    print(f"Autumn data: {len(autumn_data):,} points")
    print(f"Winter data: {len(winter_data):,} points")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot all 4 PDFs
    plt.hist(spring_data, bins=100, density=True, alpha=0.6, color='lightgreen', 
             edgecolor='black', label=f'Spring (n={len(spring_data):,})')
    plt.hist(summer_data, bins=100, density=True, alpha=0.6, color='orange', 
             edgecolor='black', label=f'Summer (n={len(summer_data):,})')
    plt.hist(autumn_data, bins=100, density=True, alpha=0.6, color='brown', 
             edgecolor='black', label=f'Autumn (n={len(autumn_data):,})')
    plt.hist(winter_data, bins=100, density=True, alpha=0.6, color='lightblue', 
             edgecolor='black', label=f'Winter (n={len(winter_data):,})')
    
    plt.title('Frequency Distribution: All Seasons', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.xlim(59.90, 60.10)  # Set consistent x-axis limits
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = ""
    if not spring_data.empty:
        stats_text += f"""Spring:
μ={spring_data.mean():.4f} Hz, σ={spring_data.std():.4f} Hz
s={stats.skew(spring_data):.4f}, κ={stats.kurtosis(spring_data, fisher=False):.4f}
"""
    if not summer_data.empty:
        stats_text += f"""Summer:
μ={summer_data.mean():.4f} Hz, σ={summer_data.std():.4f} Hz
s={stats.skew(summer_data):.4f}, κ={stats.kurtosis(summer_data, fisher=False):.4f}
"""
    if not autumn_data.empty:
        stats_text += f"""Autumn:
μ={autumn_data.mean():.4f} Hz, σ={autumn_data.std():.4f} Hz
s={stats.skew(autumn_data):.4f}, κ={stats.kurtosis(autumn_data, fisher=False):.4f}
"""
    if not winter_data.empty:
        stats_text += f"""Winter:
μ={winter_data.mean():.4f} Hz, σ={winter_data.std():.4f} Hz
s={stats.skew(winter_data):.4f}, κ={stats.kurtosis(winter_data, fisher=False):.4f}"""
    plt.text(0.02, 0.35, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=18)
    
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_all_seasons.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All seasons PDF saved to: {os.path.join(pdf_dir, 'frequency_pdf_all_seasons.png')}")
    
    # Create log-scale version for all seasons
    plt.figure(figsize=(12, 8))
    plt.hist(spring_data, bins=100, density=True, alpha=0.6, color='lightgreen', 
             edgecolor='black', label=f'Spring (n={len(spring_data):,})')
    plt.hist(summer_data, bins=100, density=True, alpha=0.6, color='orange', 
             edgecolor='black', label=f'Summer (n={len(summer_data):,})')
    plt.hist(autumn_data, bins=100, density=True, alpha=0.6, color='brown', 
             edgecolor='black', label=f'Autumn (n={len(autumn_data):,})')
    plt.hist(winter_data, bins=100, density=True, alpha=0.6, color='lightblue', 
             edgecolor='black', label=f'Winter (n={len(winter_data):,})')
    plt.yscale('log')
    plt.title('Frequency Distribution: All Seasons (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Probability Density (log scale)', fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_dir, 'frequency_pdf_all_seasons_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"All seasons PDF (log scale) saved to: {os.path.join(pdf_dir, 'frequency_pdf_all_seasons_log.png')}")

if __name__ == "__main__":
    # Load data
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'df_South_Korea_cleansed_2024-08-15_2024-12-10.pkl')
    with open(dataset_path, 'rb') as f:
        df = pd.read_pickle(f)
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
    
    # Run comparative PDF analysis
    create_comparative_pdf_plots(df, plots_dir)
    
    # Also create detailed 4-season analysis
    pdf_dir = os.path.join(plots_dir, 'pdf')
    create_detailed_seasonal_analysis(df, pdf_dir)
    
    print("\n" + "="*50)
    print("COMPARATIVE PDF ANALYSIS COMPLETE")
    print("="*50)
    print("Generated files in plots/pdf/:")
    print("Linear scale:")
    print("- frequency_pdf_summer_vs_winter.png")
    print("- frequency_pdf_day_vs_night.png")
    print("- frequency_pdf_weekday_vs_weekend.png")
    print("- frequency_pdf_all_seasons.png")
    print("Log scale:")
    print("- frequency_pdf_summer_vs_winter_log.png")
    print("- frequency_pdf_day_vs_night_log.png")
    print("- frequency_pdf_weekday_vs_weekend_log.png")
    print("- frequency_pdf_all_seasons_log.png")
