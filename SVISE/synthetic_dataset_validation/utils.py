"""
Utility functions for synthetic dataset generation and validation.

Ported from Wen et al. reference implementation:
https://github.com/KIT-IAI-DRACOS/Stochastic-modelling-of-power-grid-frequency-applied-to-islands

Adapted for the South Korean grid (60 Hz nominal frequency).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from kramersmoyal import km


# =============================================================================
# Data Cleaning (adapted from Wen's utils/Data_cleaning.py)
# =============================================================================

def data_cleaning(data, freq_limits=(59, 61), df_c=0.05, T_c=60, N_f=20):
    """Clean frequency time series by removing outliers and filling gaps.

    Args:
        data: pandas Series of frequency values
        freq_limits: (min, max) acceptable frequency range in Hz
            Default (59, 61) for 60 Hz grids; Wen used (49, 51) for 50 Hz grids.
        df_c: threshold for anomalous frequency increments (Hz)
        T_c: max allowed constant-value window length (seconds)
        N_f: max consecutive NaN values to forward-fill

    Returns:
        Cleaned pandas Series
    """
    df = data.diff()

    # Find corrupted data points
    f_too_low = np.argwhere((data < freq_limits[0]).values)[:, 0]
    f_too_high = np.argwhere((data > freq_limits[1]).values)[:, 0]
    inc_too_high = np.argwhere((df.abs() > df_c).values)[:, 0]

    print(f"Number of too high frequency values: {len(f_too_high)}")
    print(f"Number of too low frequency values: {len(f_too_low)}")
    print(f"Number of too large increments: {len(inc_too_high)}")

    # Find constant windows
    mask_const = df.abs() < 1e-9
    # Use run-length encoding to find constant windows
    changes = mask_const.ne(mask_const.shift())
    groups = changes.cumsum()
    runs = mask_const.groupby(groups).agg(['first', 'count'])
    long_const_indices = []
    cum_idx = 0
    for _, row in runs.iterrows():
        if row['first'] and row['count'] > T_c:
            long_const_indices.extend(range(cum_idx, cum_idx + int(row['count'])))
        cum_idx += int(row['count'])

    print(f"Number of windows with constant frequency for longer than {T_c}s: {len(long_const_indices) > 0}")

    # Mark corrupted data as NaN
    data_m = data.copy()
    if len(f_too_low) > 0:
        data_m.iloc[f_too_low] = np.nan
    if len(f_too_high) > 0:
        data_m.iloc[f_too_high] = np.nan
    if len(inc_too_high) > 0:
        data_m.iloc[inc_too_high] = np.nan
    if long_const_indices:
        data_m.iloc[long_const_indices] = np.nan

    # Forward-fill up to N_f values
    data_cl = data_m.ffill(limit=N_f)

    print("Clean corrupted data ...")
    return data_cl


# =============================================================================
# Gaussian Filter (from Wen's utils/Functions.py)
# =============================================================================

def data_filter(data, sigma=60):
    """Apply Gaussian smoothing to a time series.

    Used for de-trending before Kramers-Moyal analysis.
    Set sigma=0 to return the original data unchanged.

    Args:
        data: numpy array or pandas Series
        sigma: Gaussian filter width. 0 = no filtering.

    Returns:
        Filtered array
    """
    if sigma == 0:
        return np.array(data) if isinstance(data, pd.Series) else data
    return gaussian_filter1d(np.array(data), sigma=sigma)


# =============================================================================
# Integration (from Wen's utils/Functions.py)
# =============================================================================

def integrate_omega(data, time_res=1, start_value=0):
    """Integrate angular velocity to get phase angle (theta).

    Args:
        data: angular velocity array (omega, rad/s)
        time_res: time resolution in seconds
        start_value: starting index

    Returns:
        theta array (phase angle, rad), mean-centered
    """
    theta = np.zeros(len(data))
    theta[start_value] = data[start_value]
    for i in range(start_value + 1, len(data) - start_value):
        theta[i] = theta[i - 1] + time_res * data[i]
    theta = theta - np.mean(theta)
    return theta


# =============================================================================
# Daily Profile (from Wen's utils/Functions.py)
# =============================================================================

def daily_profile(data, time_res=1):
    """Compute 24-hour average load profile from the full time series.

    Averages across all days in the dataset to produce one representative
    daily profile (86400 samples at 1s resolution).

    Args:
        data: full time series array (angular velocity)
        time_res: time resolution in seconds

    Returns:
        24-hour profile array (86400 values at 1s)
    """
    n_seconds_per_day = int(24 * 3600)
    profile = np.zeros(n_seconds_per_day)
    day_number = len(data) // int(n_seconds_per_day / time_res)

    if day_number == 0:
        return profile

    for i in range(n_seconds_per_day):
        values = []
        for j in range(day_number):
            idx = int(int(i / time_res) + int(n_seconds_per_day / time_res) * j) // int(1 / time_res)
            if idx < len(data):
                values.append(data[idx])
        if values:
            profile[i] = np.mean(values)

    return profile


# =============================================================================
# Power Mismatch (from Wen's utils/Functions.py)
# =============================================================================

def power_mismatch(data, avg_for_each_hour=True, time_res=1, dispatch=1,
                   start_minute=0, end_minute=7, length_seconds_of_interval=5):
    """Estimate power mismatch Delta_P from frequency data.

    Detects rate-of-change-of-frequency (ROCOF) at dispatch boundaries
    using linear regression in short windows.

    Args:
        data: angular velocity array (omega, rad/s)
        avg_for_each_hour: if True, return array of hourly values;
            if False, return single scalar (mean absolute mismatch)
        time_res: time resolution in seconds
        dispatch: number of dispatch events per hour
            dispatch=0: no dispatch
            dispatch=1: hourly dispatch (Korea)
            dispatch=2: half-hourly dispatch
        start_minute, end_minute: window bounds around dispatch time (in minutes)
        length_seconds_of_interval: size of regression windows (seconds)

    Returns:
        Delta_P: scalar or array of power mismatch values
    """
    data_range = len(data) // (3600 * 24)
    s, e, l = 0 - start_minute, end_minute - 0, length_seconds_of_interval
    end = 2 * length_seconds_of_interval - 1
    steps = end + 1

    n_intervals = 24 * dispatch
    argm = np.zeros((n_intervals, data_range))
    Delta_P_slopes = np.zeros((n_intervals, data_range))

    for i in range(n_intervals):
        for j in range(1, data_range):
            try:
                # Find the time of maximum ROCOF in the window
                slopes_in_window = []
                for k in range(1, int((s + e) * 60 / l)):
                    start_idx = i * int(3600 / dispatch) + 3600 * 24 * j - int(s * 60) + k * l - l
                    end_idx = start_idx + steps
                    if start_idx < 0 or end_idx > len(data):
                        slopes_in_window.append(0)
                        continue
                    segment = data[start_idx:end_idx]
                    if len(segment) != steps:
                        slopes_in_window.append(0)
                        continue
                    try:
                        popt, _ = curve_fit(
                            lambda t, a, b: a + b * t,
                            np.linspace(0, end, steps),
                            segment,
                            p0=(0.0, 0.0),
                            maxfev=10000
                        )
                        slopes_in_window.append(abs(popt[1]))
                    except Exception:
                        slopes_in_window.append(0)

                if slopes_in_window:
                    argm[i, j] = np.argmax(slopes_in_window)

                # Fit at the time of maximum ROCOF
                k_best = int(argm[i, j] + 1)
                start_idx = i * int(3600 / dispatch) + 3600 * 24 * j - int(s * 60) + k_best * l - l
                end_idx = start_idx + steps
                if start_idx >= 0 and end_idx <= len(data):
                    segment = data[start_idx:end_idx]
                    if len(segment) == steps:
                        popt, _ = curve_fit(
                            lambda t, a, b: a + b * t,
                            np.linspace(0, end, steps),
                            segment,
                            p0=(0.0, 0.0),
                            maxfev=10000
                        )
                        Delta_P_slopes[i, j] = popt[1]
            except Exception:
                continue

    # Determine sign from daily profile
    sign = np.zeros(n_intervals)
    day = daily_profile(data, time_res=time_res)
    daily_prof_25 = np.zeros(25 * 3600 * time_res)
    daily_prof_25[:24 * 3600 * time_res] = day
    daily_prof_25[24 * 3600 * time_res:] = day[:1 * 3600 * time_res]

    for i in range(len(sign)):
        interval_len = int(4 / dispatch) * 900
        center = (i + 1) * interval_len
        start_idx = center - int(s * 60)
        end_idx = center + int(e * 60)
        if end_idx <= len(daily_prof_25):
            if np.mean(np.diff(daily_prof_25[start_idx:end_idx])) > 0:
                sign[(i + 1) % n_intervals] = 1
            else:
                sign[(i + 1) % n_intervals] = -1

    P_arr = np.zeros(n_intervals)
    for i in range(n_intervals):
        P_arr[i] = np.mean(np.abs(Delta_P_slopes[i, :]))

    if avg_for_each_hour:
        Delta_P = sign * P_arr
    else:
        Delta_P = np.mean(np.abs(Delta_P_slopes))

    return Delta_P


# =============================================================================
# Exponential Decay (from Wen's utils/Functions.py)
# =============================================================================

def exp_decay(data, time_res=1, size=899):
    """Estimate secondary control coefficient from exponential decay.

    Fits a*exp(-b*t) to hourly windows of the frequency data.

    Args:
        data: angular velocity array (omega)
        time_res: time resolution in seconds
        size: number of data points to fit (899 ~ 15 min at 1s)

    Returns:
        Mean decay coefficient b (c_2_decay)
    """
    steps = size + 1
    window = 3600
    data_range = len(data) // window
    c_2_decays = np.zeros(data_range)

    for j in range(1, data_range):
        start_idx = 3600 * j
        end_idx = start_idx + steps
        if end_idx > len(data):
            continue

        segment = data[start_idx:end_idx]
        t_arr = np.linspace(0, size, steps)

        try:
            # Check direction of initial transient
            if np.sum(np.diff(data[start_idx:start_idx + 10])) > 0:
                popt, _ = curve_fit(
                    lambda t, a, b: a * np.exp(-b * t),
                    t_arr, segment,
                    p0=(0.08, 0.00455),
                    maxfev=10000
                )
            else:
                popt, _ = curve_fit(
                    lambda t, a, b: (-a) * np.exp(-b * t),
                    t_arr, segment,
                    p0=(0.08, 0.00455),
                    maxfev=10000
                )
            c_2_decays[j] = popt[1]
        except Exception:
            c_2_decays[j] = 0

    # Remove outliers (top 20%)
    sorted_vals = np.sort(c_2_decays)
    trimmed = sorted_vals[:len(sorted_vals) - len(sorted_vals) // 5]
    return np.mean(trimmed)


# =============================================================================
# Kramers-Moyal Coefficient Estimation (from Wen's utils/Functions.py)
# =============================================================================

def KM_Coeff_1(data, dim=1, time_res=1, bandwidth=0.1, dist=500, order=1, start_value=0):
    """Estimate drift coefficient c_1 from Kramers-Moyal analysis.

    Uses kramersmoyal.km() to compute the first KM coefficient (drift),
    then fits a polynomial around the zero-frequency point.

    Args:
        data: angular velocity array (omega, should be de-trended)
        dim: 1 for univariate analysis, 2 for bivariate (theta, omega)
        time_res: time resolution in seconds
        bandwidth: KDE bandwidth for kramersmoyal
        dist: number of bins on each side of zero for fitting
        order: polynomial order for fit (1=linear)
        start_value: start index for integration (dim=2 only)

    Returns:
        Polynomial coefficients (for dim=1, order=1: returns array [intercept, -slope])
    """
    if dim == 1:
        powers = [0, 1, 2]
        bins = np.array([6000])
        kmc, edges = km(data, powers=powers, bins=bins, bw=bandwidth)
        zero_frequency = np.argmin(edges[0] ** 2)

        start = max(0, zero_frequency - dist)
        end = min(len(edges[0]), zero_frequency + dist)

        c = np.polyfit(edges[0][start:end], kmc[1][start:end], order)
        # Return every 2nd coefficient (matching Wen's behavior for order=1)
        c = c[::2]
        return c

    elif dim == 2:
        powers = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]])
        bins = np.array([300, 300])
        data_2d = np.array([integrate_omega(data, time_res=time_res, start_value=start_value), data])
        kmc, edges = km(data_2d.transpose(), powers=powers, bins=bins, bw=bandwidth)

        def f_0_1(x, p_1, c_2):
            return p_1 * x[1] + c_2 * x[0]

        zero_angle = np.argmin(edges[0] ** 2)
        zero_frequency = np.argmin(edges[1] ** 2)

        if isinstance(dist, (list, tuple)):
            d0, d1 = dist
        else:
            d0 = d1 = dist

        side_x = edges[0][zero_angle - d0:zero_angle + d0]
        side_y = edges[1][zero_frequency - d1:zero_frequency + d1]
        X1, X2 = np.meshgrid(side_x, side_y)
        size = X1.shape
        x1_1d = X1.reshape((1, np.prod(size)))
        x2_1d = X2.reshape((1, np.prod(size)))
        xdata = np.vstack((x1_1d, x2_1d))
        z = np.array([[kmc[2, zero_angle - d0 + i, zero_frequency - d1 + j] / time_res
                        for i in range(2 * d0)]
                       for j in range(2 * d1)])
        Z = z.reshape(np.prod(size))
        popt, _ = curve_fit(f_0_1, xdata, Z)
        return popt


def KM_Coeff_2(data, dim=1, time_res=1, bandwidth=0.1, dist=500, multiplicative_noise=True, start_value=0):
    """Estimate diffusion coefficient (noise amplitude) from Kramers-Moyal analysis.

    Args:
        data: angular velocity array (omega, should be de-trended)
        dim: 1 for univariate, 2 for bivariate
        time_res: time resolution in seconds
        bandwidth: KDE bandwidth for kramersmoyal
        dist: number of bins on each side of zero
        multiplicative_noise: if False, return constant additive noise epsilon;
            if True, return (d_2, d_0) for state-dependent noise
        start_value: start index for integration (dim=2 only)

    Returns:
        epsilon (scalar for additive) or (d_2, d_0) tuple for multiplicative
    """
    if dim == 1:
        powers = [0, 1, 2]
        bins = np.array([6000])
        kmc, edges = km(data, powers=powers, bins=bins, bw=bandwidth)
        zero_frequency = np.argmin(edges[0] ** 2)

        start = max(0, zero_frequency - dist)
        end = min(len(edges[0]), zero_frequency + dist)

        if not multiplicative_noise:
            epsilon = np.sqrt(2 * np.mean(kmc[2, start:end]))
            return epsilon
        else:
            peak = start + np.argmin(kmc[2, start:end])
            try:
                d_2 = curve_fit(
                    lambda t, a: a * (t - 0) ** 2 + kmc[2, zero_frequency],
                    edges[0][start:end],
                    kmc[2, peak - dist:peak + dist]
                )[0]
            except Exception:
                d_2 = np.array([0.0])
            diff_zero = kmc[2, peak]
            d_0 = diff_zero
            return (d_2[0], d_0)

    elif dim == 2:
        powers = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2]])
        bins = np.array([300, 300])
        data_2d = np.array([integrate_omega(data, time_res=time_res, start_value=start_value), data])
        kmc, edges = km(data_2d.transpose(), powers=powers, bins=bins, bw=bandwidth)

        zero_angle = np.argmin(edges[0] ** 2)
        zero_frequency = np.argmin(edges[1] ** 2)

        if isinstance(dist, (list, tuple)):
            d0, d1 = dist
        else:
            d0 = d1 = dist

        if not multiplicative_noise:
            epsilon = np.sqrt(2 * np.mean(
                kmc[5, zero_angle - d0:zero_angle + d0,
                    zero_frequency - d1:zero_frequency + d1] / time_res
            ))
            return epsilon
        else:
            def f_0_2(x, a, b):
                return a * (x[1]) ** 2 + b

            side_x = edges[0][zero_angle - d0:zero_angle + d0]
            side_y = edges[1][zero_frequency - d1:zero_frequency + d1]
            X1, X2 = np.meshgrid(side_x, side_y)
            size = X1.shape
            x1_1d = X1.reshape((1, np.prod(size)))
            x2_1d = X2.reshape((1, np.prod(size)))
            xdata = np.vstack((x1_1d, x2_1d))
            z = np.array([[kmc[5, zero_angle - d0 + i, zero_frequency - d1 + j] / time_res
                            for i in range(2 * d0)]
                           for j in range(2 * d1)])
            Z = z.reshape(np.prod(size))
            popt, _ = curve_fit(f_0_2, xdata, Z)
            return popt


# =============================================================================
# Euler-Maruyama SDE Integration - Model 2 (from Wen's utils/Functions.py)
# =============================================================================

def Euler_Maruyama_Model2(data, c_1, c_2_decay, Delta_P, epsilon=0,
                          time_res=1, dispatch=1, delta_t=1.0,
                          t_final=2592000, seed=42):
    """Generate synthetic frequency data using Model 2 (linear with dispatch).

    Model 2 equations:
        dtheta/dt = omega
        domega/dt = c_1*omega + c_2_decay*c_1*theta + Delta_P*P(t)*sign(t) + epsilon*dW

    Args:
        data: real frequency data (omega, for daily profile estimation)
        c_1: drift coefficient (negative, e.g., -0.0295)
        c_2_decay: secondary control decay rate
        Delta_P: mean power mismatch scalar (from power_mismatch(avg_for_each_hour=False))
        epsilon: noise amplitude. Set to 0 for noiseless.
        time_res: time resolution of data in seconds
        dispatch: dispatch events per hour (0=none, 1=hourly, 2=half-hourly)
        delta_t: Euler-Maruyama time step in seconds
        t_final: total simulation duration in seconds (default: 30 days = 2592000)
        seed: random seed for reproducibility

    Returns:
        omega: angular velocity array
        theta: phase angle array
        P_driving: the effective driving force Delta_P*P(t)*sign(t) at each step
    """
    np.random.seed(seed)

    t_steps = int(t_final / delta_t)
    time = np.linspace(0.0, t_final, t_steps)

    omega = np.zeros(t_steps)
    theta = np.zeros(t_steps)
    theta[0] = np.random.normal() / 10
    omega[0] = np.random.normal() / 10

    dW = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=t_steps)

    P = np.ones(t_steps)
    sign_P = np.zeros(t_steps)
    P_driving = np.zeros(t_steps)

    for i in range(1, t_steps):
        if dispatch != 0:
            # Sign alternates every 12 hours
            if i % (12 * 3600 / delta_t) < 6 * 3600 / delta_t:
                sign_P[i] = 1
            else:
                sign_P[i] = -1

            # Dispatch intensity pattern
            if i % (60 * 60 / delta_t) < (4 / dispatch) * 15 * 60 / delta_t:
                P[i] = 1
            else:
                P[i] = 1 / 3

        driving_force = Delta_P * P[i] * sign_P[i]
        P_driving[i] = driving_force

        theta[i] = theta[i - 1] + delta_t * omega[i - 1]
        omega[i] = (omega[i - 1]
                     + delta_t * (c_1 * omega[i - 1]
                                  + c_2_decay * c_1 * theta[i - 1]
                                  + driving_force)
                     + epsilon * dW[i])

    return omega, theta, P_driving
