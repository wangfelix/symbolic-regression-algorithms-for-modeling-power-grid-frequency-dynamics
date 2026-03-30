"""
SDE Simulation for PDF & ACF Comparison.

Aggregates per-chunk drift coefficients and diffusion from SVISE results,
simulates long SDE trajectories via Euler-Maruyama, and compares the
resulting PDF and ACF of omega against empirical data.

Models:
  - SVISE (B): degree-2 polynomial drift (combo38)
  - SVISE (C): degree-3 polynomial drift (combo5)
  - 1D-L-KM:  linear Kramers-Moyal reference model

Usage:
    python simulate_sde_pdf_acf.py
"""

import os
import sys
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from numpy.fft import rfft, irfft

# Sympy for parsing SVISE equation strings
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# =============================================================================
# Constants
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# Simulation parameters
DT = 1.0                    # seconds
T_FINAL = 30 * 86400        # 30 days
BURNIN = 86400               # 1 day burn-in
SEED = 42
ACF_MAX_LAG = 7200           # 120 minutes at 1s

# 1D-L-KM reference model parameters (from Kramers-Moyal estimation)
KM_C_OMEGA = -0.00906
KM_C_THETA = -1.53e-5
KM_EPSILON = 0.0109          # noise amplitude

# Data paths
PARQUET_PATH = os.path.join(SCRIPT_DIR, "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
SVISE_B_CSV = os.path.join(SCRIPT_DIR, "results_5min_all_chunks/run_SLURM_3733209_combo38/all_chunks_combined.csv")
SVISE_C_CSV = os.path.join(SCRIPT_DIR, "results_5min_all_chunks/run_SLURM_3733210_combo5/all_chunks_combined.csv")

# Coefficient order:
# [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
# = [const, theta, omega, theta^2, theta_omega, omega^2,
#    theta^3, theta^2_omega, theta_omega^2, omega^3]
TERM_NAMES = [
    "const", "theta", "omega", "theta^2", "theta_omega",
    "omega^2", "theta^3", "theta^2_omega", "theta_omega^2", "omega^3",
]

# Plot colors
COLORS = {
    "Empirical": "black",
    "SVISE (B)": "#2ca089",
    "SVISE (C)": "#e07b39",
    "1D-L-KM": "#3b9dd4",
}

# =============================================================================
# Sympy setup for equation parsing (reused from analyse_omega_coefficients.py)
# =============================================================================
_SYMPY_TRANSFORMS = (standard_transformations + (implicit_multiplication_application, convert_xor))
_THETA, _OMEGA = sympy.symbols('theta omega')
_X0, _X1 = sympy.symbols('x0 x1')
_GLOBAL_DICT = {
    'theta': _THETA, 'omega': _OMEGA, 'x0': _X0, 'x1': _X1,
    'Symbol': sympy.Symbol, 'Float': sympy.Float, 'Integer': sympy.Integer,
    'Add': sympy.Add, 'Mul': sympy.Mul, 'Pow': sympy.Pow,
}


@functools.lru_cache(maxsize=100000)
def parse_omega_equation(eq_str):
    """Parse a physical-space omega equation string and extract 10 polynomial coefficients."""
    if not isinstance(eq_str, str) or "nan" in eq_str.lower() or "error" in eq_str.lower() or eq_str == "N/A":
        return None
    try:
        expr = parse_expr(eq_str, transformations=_SYMPY_TRANSFORMS, global_dict=_GLOBAL_DICT)
        expr = sympy.expand(expr)
        expr = expr.subs({_X0: _THETA, _X1: _OMEGA})

        c0 = float(expr.subs({_THETA: 0, _OMEGA: 0}))
        c1 = float(expr.coeff(_THETA, 1).subs({_OMEGA: 0}))
        c2 = float(expr.coeff(_OMEGA, 1).subs({_THETA: 0}))
        c3 = float(expr.coeff(_THETA, 2).subs({_OMEGA: 0}))
        c4 = float(expr.coeff(_THETA * _OMEGA))
        c5 = float(expr.coeff(_OMEGA, 2).subs({_THETA: 0}))
        c6 = float(expr.coeff(_THETA, 3).subs({_OMEGA: 0}))
        c7 = float(expr.coeff(_THETA**2 * _OMEGA))
        c8 = float(expr.coeff(_THETA * _OMEGA**2))
        c9 = float(expr.coeff(_OMEGA, 3).subs({_THETA: 0}))
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    except Exception:
        return None


# =============================================================================
# Coefficient loading
# =============================================================================

def load_svise_coefficients(csv_path):
    """Load SVISE results CSV and return median coefficients + median diffusion.

    Returns:
        (coeffs, diffusion): coeffs is np.array[10], diffusion is float
    """
    print(f"Loading SVISE results from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Parse all physical-space omega equations
    all_coeffs = []
    for eq_str in df["Eq_Omega_Physical"]:
        c = parse_omega_equation(str(eq_str))
        if c is not None:
            all_coeffs.append(c)

    coeffs_array = np.array(all_coeffs)
    print(f"  Valid equations: {len(coeffs_array)}")

    median_coeffs = np.nanmedian(coeffs_array, axis=0)

    # Diffusion
    diff_col = pd.to_numeric(df["Diffusion_Omega"], errors="coerce")
    median_diff = float(diff_col.median())
    print(f"  Median Diffusion_Omega: {median_diff:.6e}")

    return median_coeffs, median_diff


def get_1dlkm_coefficients():
    """Return 1D-L-KM reference model coefficients and diffusion.

    Returns:
        (coeffs, diffusion): coeffs is np.array[10], diffusion is float
    """
    coeffs = np.zeros(10)
    coeffs[1] = KM_C_THETA   # theta
    coeffs[2] = KM_C_OMEGA   # omega
    diffusion = KM_EPSILON ** 2  # sigma^2
    return coeffs, diffusion


# =============================================================================
# Euler-Maruyama SDE simulation
# =============================================================================

import ctypes as _ct

# Load compiled C loop (compiled from _em_loop.c with: gcc -O3 -shared -fPIC -o _em_loop.so _em_loop.c -lm)
_SO_PATH = os.path.join(SCRIPT_DIR, "_em_loop.so")
try:
    _lib = _ct.CDLL(_SO_PATH)
    _lib.em_loop.restype = _ct.c_int
    _lib.em_loop.argtypes = [
        _ct.POINTER(_ct.c_double),  # omega
        _ct.POINTER(_ct.c_double),  # theta
        _ct.POINTER(_ct.c_double),  # dW
        _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double,  # c0-c4
        _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double, _ct.c_double,  # c5-c9
        _ct.c_double, _ct.c_double, _ct.c_int,  # sigma, dt, n_steps
    ]
    _EM_BACKEND = "C"
except OSError:
    _lib = None
    _EM_BACKEND = "python"


def _em_loop(omega, theta, dW, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, sigma, dt):
    """Euler-Maruyama loop — dispatches to C if available, else pure Python."""
    n_steps = len(omega)
    if _lib is not None:
        return _lib.em_loop(
            omega.ctypes.data_as(_ct.POINTER(_ct.c_double)),
            theta.ctypes.data_as(_ct.POINTER(_ct.c_double)),
            dW.ctypes.data_as(_ct.POINTER(_ct.c_double)),
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,
            sigma, dt, n_steps,
        )
    # Fallback: pure Python
    n_clamp = 0
    for i in range(1, n_steps):
        th = theta[i - 1]
        om = omega[i - 1]
        f = c0 + th * (c1 + th * (c3 + th * c6)) \
            + om * (c2 + th * (c4 + th * c7) + om * (c5 + th * c8 + om * c9))
        theta[i] = th + om * dt
        omega[i] = om + f * dt + sigma * dW[i]
        if omega[i] > 2.0:
            omega[i] = 2.0
            n_clamp += 1
        elif omega[i] < -2.0:
            omega[i] = -2.0
            n_clamp += 1
        if theta[i] > 100.0:
            theta[i] = 100.0
            n_clamp += 1
        elif theta[i] < -100.0:
            theta[i] = -100.0
            n_clamp += 1
    return n_clamp


def simulate_sde(coeffs, diffusion, dt=DT, t_final=T_FINAL, seed=SEED):
    """Simulate 2D SDE via Euler-Maruyama.

    SDE:  dtheta = omega * dt
          domega = f(theta, omega) * dt + sigma * dW
    where f is a polynomial with the given coefficients,
    sigma = sqrt(diffusion), and dW ~ N(0, sqrt(dt)).

    Args:
        coeffs: array of 10 polynomial coefficients [c0..c9]
        diffusion: sigma^2 (variance per unit time)
        dt: time step in seconds
        t_final: total simulation time in seconds
        seed: random seed

    Returns:
        (omega, theta) arrays after burn-in
    """
    print(f"    EM backend: {_EM_BACKEND}")
    np.random.seed(seed)
    n_steps = int(t_final / dt)
    sigma = np.sqrt(diffusion)

    omega = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    omega[0] = np.random.normal() * 0.01
    theta[0] = np.random.normal() * 0.01

    dW = np.random.normal(0, np.sqrt(dt), size=n_steps)

    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs

    n_clamp = _em_loop(omega, theta, dW, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, sigma, dt)

    if n_clamp > 0:
        print(f"  WARNING: {n_clamp} clamp events during simulation")

    # Discard burn-in
    burnin_steps = int(BURNIN / dt)
    return omega[burnin_steps:], theta[burnin_steps:]


# =============================================================================
# PDF computation
# =============================================================================

def compute_pdf(data, bins=500, range_sigma=5):
    """Compute histogram-based PDF.

    Returns:
        (bin_centers, density)
    """
    mu = np.mean(data)
    sigma = np.std(data)
    lo = mu - range_sigma * sigma
    hi = mu + range_sigma * sigma
    density, bin_edges = np.histogram(data, bins=bins, range=(lo, hi), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, density


# =============================================================================
# ACF computation (FFT-based)
# =============================================================================

def compute_acf(data, max_lag=ACF_MAX_LAG, max_samples=None):
    """Compute autocorrelation function via FFT.

    Args:
        data: 1D array
        max_lag: maximum lag in samples
        max_samples: if set, use only the first N samples (for speed on large arrays)

    Returns:
        acf array of length max_lag+1, normalized so acf[0]=1.
    """
    if max_samples is not None and len(data) > max_samples:
        data = data[:max_samples]
    n = len(data)
    x = data - np.mean(data)
    # Zero-pad to next power of 2 for FFT efficiency
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = rfft(x, n=nfft)
    acf_full = irfft(f * np.conj(f), n=nfft)[:n]
    acf_full /= acf_full[0]
    return acf_full[:max_lag + 1]


# =============================================================================
# Empirical data loading
# =============================================================================

def load_empirical_omega():
    """Load empirical frequency data and return omega array."""
    print(f"Loading empirical data from {os.path.basename(PARQUET_PATH)}...")
    data = pd.read_parquet(PARQUET_PATH)
    data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(subset=['freq', 'QI'])
    freq = data_filtered['freq'].values
    omega = (freq - 60.0) * 2 * np.pi
    print(f"  Valid samples: {len(omega):,}")
    return omega


# =============================================================================
# Plotting: per-model PDF (2x2 grid)
# =============================================================================

def plot_pdf_single(omega, label, color, output_path):
    """Plot PDF analysis for a single model (2x2 grid: hist, log-hist, Q-Q, box)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Convert to Hz deviation for readability
    freq_dev = omega / (2 * np.pi)  # Hz deviation from 60 Hz

    # Panel 1: Histogram (linear)
    ax = axes[0, 0]
    ax.hist(freq_dev, bins=100, density=True, alpha=0.7, color=color, edgecolor='black', linewidth=0.3)
    ax.set_title(f'{label} — Frequency Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency deviation (Hz)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    # Panel 2: Histogram (log scale)
    ax = axes[0, 1]
    ax.hist(freq_dev, bins=100, density=True, alpha=0.7, color=color, edgecolor='black', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_title(f'{label} — Log Scale', fontsize=12, fontweight='bold')
    ax.set_xlabel('Frequency deviation (Hz)')
    ax.set_ylabel('Density (log)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Q-Q plot
    ax = axes[1, 0]
    stats.probplot(freq_dev, dist="norm", plot=ax)
    ax.set_title(f'{label} — Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 4: Box plot
    ax = axes[1, 1]
    ax.boxplot(freq_dev, vert=True)
    ax.set_title(f'{label} — Box Plot', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency deviation (Hz)')
    ax.grid(True, alpha=0.3)

    # Stats annotation
    mu_hz = np.mean(freq_dev)
    std_hz = np.std(freq_dev)
    skew = float(stats.skew(freq_dev))
    kurt = float(stats.kurtosis(freq_dev, fisher=False))
    fig.suptitle(
        f'{label}:  $\\mu$={mu_hz:.4f} Hz,  $\\sigma$={std_hz:.4f} Hz,  '
        f'skew={skew:.3f},  $\\kappa$={kurt:.3f}',
        fontsize=11, y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Plotting: per-model ACF
# =============================================================================

def plot_acf_single(acf, label, color, output_path):
    """Plot ACF for a single model."""
    lags_minutes = np.arange(len(acf)) / 60.0

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(lags_minutes, acf, color=color, linewidth=1)
    ax.set_title(f'{label} — Autocorrelation Function', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('ACF')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlim(0, lags_minutes[-1])

    # Annotate key values
    key_lags_min = [1, 5, 10, 30, 60, 120]
    for lag_min in key_lags_min:
        idx = lag_min * 60
        if idx < len(acf):
            ax.annotate(f'{acf[idx]:.3f}', xy=(lag_min, acf[idx]),
                        fontsize=8, ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Plotting: comparison PDF
# =============================================================================

def plot_pdf_comparison(empirical_omega, models, output_path):
    """Overlay PDF of all models + empirical on 2 panels (linear + log)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Compute empirical PDF
    emp_centers, emp_density = compute_pdf(empirical_omega)

    for ax, log_scale in [(ax1, False), (ax2, True)]:
        ax.plot(emp_centers, emp_density, color=COLORS["Empirical"],
                linewidth=2, label="Empirical", zorder=10)

        for m in models:
            centers, density = compute_pdf(m["omega"])
            ax.plot(centers, density, color=m["color"],
                    linewidth=1.2, label=m["name"], alpha=0.85)

        if log_scale:
            ax.set_yscale('log')
            ax.set_title('PDF Comparison (log scale)', fontsize=12, fontweight='bold')
        else:
            ax.set_title('PDF Comparison (linear scale)', fontsize=12, fontweight='bold')

        ax.set_xlabel('$\\omega$ (rad/s)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Plotting: comparison ACF
# =============================================================================

def plot_acf_comparison(empirical_acf, models, output_path):
    """Overlay ACF of all models + empirical."""
    fig, ax = plt.subplots(figsize=(12, 6))

    lags_minutes = np.arange(len(empirical_acf)) / 60.0
    ax.plot(lags_minutes, empirical_acf, color=COLORS["Empirical"],
            linewidth=2, label="Empirical", zorder=10)

    for m in models:
        lags_m = np.arange(len(m["acf"])) / 60.0
        ax.plot(lags_m, m["acf"], color=m["color"],
                linewidth=1.2, label=m["name"], alpha=0.85)

    ax.set_title('ACF Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag (minutes)')
    ax.set_ylabel('ACF')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlim(0, ACF_MAX_LAG / 60.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- 1. Load coefficients ----
    print("\n" + "=" * 60)
    print("LOADING MODEL COEFFICIENTS")
    print("=" * 60)

    svise_b_coeffs, svise_b_diff = load_svise_coefficients(SVISE_B_CSV)
    svise_c_coeffs, svise_c_diff = load_svise_coefficients(SVISE_C_CSV)
    km_coeffs, km_diff = get_1dlkm_coefficients()

    # Print coefficient table
    print("\n" + "-" * 80)
    print(f"{'Term':<16} {'SVISE (B)':>12} {'SVISE (C)':>12} {'1D-L-KM':>12}")
    print("-" * 80)
    for i, name in enumerate(TERM_NAMES):
        print(f"{name:<16} {svise_b_coeffs[i]:>12.6e} {svise_c_coeffs[i]:>12.6e} {km_coeffs[i]:>12.6e}")
    print("-" * 80)
    print(f"{'Diffusion':<16} {svise_b_diff:>12.6e} {svise_c_diff:>12.6e} {km_diff:>12.6e}")
    print(f"{'sigma':<16} {np.sqrt(svise_b_diff):>12.6e} {np.sqrt(svise_c_diff):>12.6e} {np.sqrt(km_diff):>12.6e}")
    print("-" * 80)

    # ---- 2. Define models ----
    model_configs = [
        {"name": "SVISE (B)", "coeffs": svise_b_coeffs, "diffusion": svise_b_diff,
         "color": COLORS["SVISE (B)"], "file_tag": "svise_b"},
        {"name": "SVISE (C)", "coeffs": svise_c_coeffs, "diffusion": svise_c_diff,
         "color": COLORS["SVISE (C)"], "file_tag": "svise_c"},
        {"name": "1D-L-KM", "coeffs": km_coeffs, "diffusion": km_diff,
         "color": COLORS["1D-L-KM"], "file_tag": "1dlkm"},
    ]

    # ---- 3. Simulate SDEs ----
    print("\n" + "=" * 60)
    print("SIMULATING SDEs (30 days each)")
    print("=" * 60)

    for m in model_configs:
        print(f"\n  Simulating {m['name']}...")
        omega_sim, theta_sim = simulate_sde(m["coeffs"], m["diffusion"])
        m["omega"] = omega_sim
        m["theta"] = theta_sim
        print(f"    Samples after burn-in: {len(omega_sim):,}")
        print(f"    omega: mean={np.mean(omega_sim):.6e}, std={np.std(omega_sim):.6e}")

    # ---- 4. Per-model plots ----
    print("\n" + "=" * 60)
    print("GENERATING PER-MODEL PLOTS")
    print("=" * 60)

    for m in model_configs:
        tag = m["file_tag"]
        plot_pdf_single(
            m["omega"], m["name"], m["color"],
            os.path.join(FIGURES_DIR, f"sde_pdf_{tag}.png"),
        )
        print(f"  Computing ACF for {m['name']}...")
        m["acf"] = compute_acf(m["omega"])
        plot_acf_single(
            m["acf"], m["name"], m["color"],
            os.path.join(FIGURES_DIR, f"sde_acf_{tag}.png"),
        )

    # ---- 5. Load empirical data ----
    print("\n" + "=" * 60)
    print("LOADING EMPIRICAL DATA")
    print("=" * 60)

    empirical_omega = load_empirical_omega()

    print("  Computing empirical ACF (using first 30 days for speed)...")
    empirical_acf = compute_acf(empirical_omega, max_samples=30 * 86400)

    # Per-model PDF/ACF for empirical too
    plot_pdf_single(
        empirical_omega, "Empirical", COLORS["Empirical"],
        os.path.join(FIGURES_DIR, "sde_pdf_empirical.png"),
    )
    plot_acf_single(
        empirical_acf, "Empirical", COLORS["Empirical"],
        os.path.join(FIGURES_DIR, "sde_acf_empirical.png"),
    )

    # ---- 6. Comparison plots ----
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)

    plot_pdf_comparison(
        empirical_omega, model_configs,
        os.path.join(FIGURES_DIR, "sde_pdf_comparison.png"),
    )
    plot_acf_comparison(
        empirical_acf, model_configs,
        os.path.join(FIGURES_DIR, "sde_acf_comparison.png"),
    )

    # ---- 7. Print summary statistics ----
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    emp_freq_dev = empirical_omega / (2 * np.pi)
    print(f"\n{'Model':<14} {'mean(Hz dev)':>12} {'std(Hz dev)':>12} {'skew':>8} {'kurtosis':>8}")
    print("-" * 60)
    print(f"{'Empirical':<14} {np.mean(emp_freq_dev):>12.6f} {np.std(emp_freq_dev):>12.6f} "
          f"{float(stats.skew(emp_freq_dev)):>8.3f} {float(stats.kurtosis(emp_freq_dev, fisher=False)):>8.3f}")

    for m in model_configs:
        freq_dev = m["omega"] / (2 * np.pi)
        print(f"{m['name']:<14} {np.mean(freq_dev):>12.6f} {np.std(freq_dev):>12.6f} "
              f"{float(stats.skew(freq_dev)):>8.3f} {float(stats.kurtosis(freq_dev, fisher=False)):>8.3f}")

    print("\nACF at key lags:")
    print(f"{'Model':<14} {'1min':>8} {'5min':>8} {'10min':>8} {'30min':>8} {'60min':>8} {'120min':>8}")
    print("-" * 70)
    for label, acf_data in [("Empirical", empirical_acf)] + [(m["name"], m["acf"]) for m in model_configs]:
        vals = []
        for lag_min in [1, 5, 10, 30, 60, 120]:
            idx = lag_min * 60
            vals.append(f"{acf_data[idx]:>8.4f}" if idx < len(acf_data) else f"{'N/A':>8}")
        print(f"{label:<14} {''.join(vals)}")

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
