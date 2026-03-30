import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy import stats


def _timestamp():
    """Return current timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def autocorr_func(x, max_lag):
    """Calculate autocorrelation function properly"""
    n = len(x)
    autocorr = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            x1 = x[:-lag]
            x2 = x[lag:]
            if len(x1) > 0 and len(x2) > 0:
                corr = np.corrcoef(x1, x2)[0, 1]
                autocorr[lag] = corr if not np.isnan(corr) else 0.0
            else:
                autocorr[lag] = 0.0

    return autocorr


def calculate_hurst_dfa(x, min_window=10,
                        max_window=None,
                        num_scales=30):
    """
    Calculate the Hurst exponent via Detrended
    Fluctuation Analysis (DFA-1), matching the method
    used by Oberhofer et al. (2025).

    DFA-1 uses linear detrending within each window.
    The Hurst exponent equals the slope of
    log(F(n)) vs log(n).

    Parameters
    ----------
    x : array-like
        Time series data.
    min_window : int
        Smallest window size (default 10).
    max_window : int or None
        Largest window size (default N // 4).
    num_scales : int
        Number of log-spaced window sizes to evaluate.

    Returns
    -------
    H : float
        Hurst exponent (DFA-1 scaling exponent).
    r_squared : float
        R-squared of the log-log fit.
    p_value : float
        P-value of the log-log regression.
    """
    n = len(x)
    if n < 20:
        return np.nan, np.nan, np.nan

    if max_window is None:
        max_window = n // 4

    # Step 1: cumulative sum of mean-centered series
    y = np.cumsum(x - np.mean(x))

    # Step 2: generate log-spaced window sizes
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num_scales
        ).astype(int)
    )

    fluctuations = []
    valid_windows = []

    for w in window_sizes:
        n_windows = n // w
        if n_windows < 2:
            continue

        rms_list = []
        for i in range(n_windows):
            segment = y[i * w:(i + 1) * w]
            t = np.arange(w, dtype=np.float64)
            # Linear detrend (DFA-1)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            rms_list.append(
                np.sqrt(np.mean((segment - trend) ** 2))
            )

        if len(rms_list) > 0:
            fluctuations.append(np.mean(rms_list))
            valid_windows.append(w)

    if len(valid_windows) < 3:
        return np.nan, np.nan, np.nan

    ws = np.array(valid_windows, dtype=np.float64)
    fs = np.array(fluctuations, dtype=np.float64)

    # Step 3: log-log regression
    log_ws = np.log(ws)
    log_fs = np.log(fs)

    slope, intercept, r_value, p_value, std_err = \
        stats.linregress(log_ws, log_fs)

    # For DFA-1 on stationary processes:
    # slope = Hurst exponent H
    return slope, r_value ** 2, p_value


def create_acf_plot(df, plots_dir, max_lag_hours=2):
    """
    Create ACF plot using all data, showing lags up
    to max_lag_hours. Also computes the Hurst exponent
    via DFA on the full dataset.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'freq' column.
    plots_dir : str
        Directory to save plots.
    max_lag_hours : int
        Maximum lag to show in hours (default 2).
    """
    print("Creating ACF plot using ALL data...")

    freq_autocorr = df['freq'].dropna().values
    print(
        f"Using ALL data: {len(freq_autocorr):,} points "
        f"for ACF calculation"
    )
    print(
        f"Data spans: {df.index.min()} to {df.index.max()}"
    )
    duration_hours = (
        (df.index.max() - df.index.min()).total_seconds()
        / 3600
    )
    print(f"Total duration: {duration_hours:.1f} hours")

    # Calculate autocorrelation up to max_lag_hours
    max_lag_seconds = max_lag_hours * 3600
    autocorr = autocorr_func(freq_autocorr, max_lag_seconds)

    # Create time arrays
    dt = 1.0  # 1 second sampling
    lags = np.arange(len(autocorr)) * dt
    lags_minutes = lags / 60

    # ---- Plot 1: ACF ----
    plt.figure(figsize=(15, 6))
    plt.plot(lags_minutes, autocorr, 'b-', linewidth=1)
    plt.title(
        'Autocorrelation Function',
        fontsize=14, fontweight='bold'
    )
    plt.xlabel('Lag (minutes)')
    plt.ylabel('Correlation')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    acf_path = os.path.join(
        plots_dir,
        f'frequency_acf_2hour_analysis_{_timestamp()}.png'
    )
    plt.savefig(acf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ACF plot saved to: {acf_path}")

    # ---- Plot 2: time series sample ----
    plt.figure(figsize=(15, 6))
    sample_start = 16 * 3600  # 4 PM
    sample_data = df['freq'].iloc[
        sample_start:sample_start + 7200
    ]
    time_hours = (
        np.arange(len(sample_data)) / 3600 + 16
    )
    plt.plot(
        time_hours, sample_data.values,
        linewidth=0.8, alpha=0.8
    )
    plt.title(
        'Frequency Time Series',
        fontsize=14, fontweight='bold'
    )
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ts_path = os.path.join(
        plots_dir,
        f'frequency_timeseries_2hour_sample_{_timestamp()}.png'
    )
    plt.savefig(ts_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to: {ts_path}")

    # ---- ACF statistics at key lags ----
    print(
        f"\nACF values at key time intervals "
        f"(using {len(freq_autocorr):,} points):"
    )
    key_lags_minutes = [1, 5, 10, 30, 60, 120]
    key_lags_seconds = [m * 60 for m in key_lags_minutes]
    for lag_sec, lag_min in zip(
        key_lags_seconds, key_lags_minutes
    ):
        if lag_sec < len(autocorr):
            print(
                f"  Lag {lag_sec:5d}s ({lag_min:3d} min): "
                f"{autocorr[lag_sec]:.6f}"
            )

    # Significance at 1h and 2h
    threshold = 2 / np.sqrt(len(freq_autocorr))
    print(f"\nSignificance threshold (95%): {threshold:.8f}")
    for lag_sec, lag_min in [(3600, 60), (7200, 120)]:
        if lag_sec < len(autocorr):
            sig = abs(autocorr[lag_sec]) > threshold
            print(
                f"  Lag {lag_sec}s ({lag_min} min): "
                f"ACF={autocorr[lag_sec]:.6f}, "
                f"Significant={sig}"
            )

    # ---- Hurst exponent via DFA (full dataset) ----
    print(f"\n{'=' * 50}")
    print("HURST EXPONENT ANALYSIS (DFA-1)")
    print("=" * 50)
    print(
        "Calculating Hurst exponent using Detrended "
        "Fluctuation Analysis on the FULL dataset..."
    )
    print(f"Number of data points: {len(freq_autocorr):,}")

    hurst, r_sq, p_val = calculate_hurst_dfa(
        freq_autocorr
    )

    if not np.isnan(hurst):
        print(f"Hurst Exponent (H): {hurst:.6f}")
        print(f"R-squared:          {r_sq:.6f}")
        print(f"P-value:            {p_val:.2e}")

        if hurst > 0.5:
            print(
                f"Interpretation: H = {hurst:.3f} > 0.5 "
                f"-> Long-range positive correlation "
                f"(persistent)"
            )
        elif hurst < 0.5:
            print(
                f"Interpretation: H = {hurst:.3f} < 0.5 "
                f"-> Long-range negative correlation "
                f"(anti-persistent)"
            )
        else:
            print(
                f"Interpretation: H = {hurst:.3f} = 0.5 "
                f"-> No long-range correlation "
                f"(random walk)"
            )
    else:
        print(
            "Could not calculate Hurst exponent "
            "(insufficient data or calculation error)"
        )

    return autocorr, lags_minutes


if __name__ == "__main__":
    # Standalone: use the extended dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'dataset',
        'South_Korea_2024-08-15_2025-08-31_1s.pkl'
    )
    with open(dataset_path, 'rb') as f:
        df = pd.read_pickle(f)

    plots_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'plots'
    )
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    autocorr, lags = create_acf_plot(df, plots_dir)