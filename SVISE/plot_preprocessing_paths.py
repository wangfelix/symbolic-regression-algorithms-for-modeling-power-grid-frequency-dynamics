"""
Paper figure: two preprocessing paths (SINDy/PySR vs SVISE) using real data from chunk 93070.

Row 1: Empirical -> Gaussian smoothed -> SINDy forward sim
Row 2: Empirical -> SVISE forward sim (with diffusion band)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint
import warnings
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_IDX = 93070
SIGMA = 15
T_SCALE = 30.0
DT = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
SVISE_CSV = os.path.join(SCRIPT_DIR, "results_5min_all_chunks/run_SLURM_3733210_combo5/all_chunks_combined.csv")
SINDY_CSV = os.path.join(SCRIPT_DIR, "results_sindy_5min_all_chunks/run_SLURM_3753891_sindy/all_chunks_combined.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures")


# ── Data Loading (from plot_forward_sim_vs_empirical.py) ───────────────────────

def load_data(data_path, limit_interpolation=10):
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)
    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(method='time', limit=limit_interpolation)
    return data


def get_all_valid_chunks(data):
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & (data['freq'].notna())].dropna(subset=['freq', 'QI'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])
    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))
    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == 300:
            valid_chunks.append((chunk_start, group))
    return valid_chunks


# ── SVISE equation parsing & simulation (from plot_forward_sim_vs_empirical.py) ─

def parse_equation_svise(eq_str):
    coeffs = {
        "1": 0.0, "theta": 0.0, "omega": 0.0,
        "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0,
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }
    eq_str = eq_str.replace("+ -", "+-").replace("- ", "-").replace("  ", " ").strip()
    terms = []
    current_term = ""
    for char in eq_str:
        if char == '+' and current_term.strip():
            terms.append(current_term.strip())
            current_term = ""
        else:
            current_term += char
    if current_term.strip():
        terms.append(current_term.strip())
    term_patterns = [
        ("theta^3", "theta^3"), ("theta^2 omega", "theta^2 omega"),
        ("theta omega^2", "theta omega^2"), ("omega^3", "omega^3"),
        ("theta^2", "theta^2"), ("theta omega", "theta omega"),
        ("omega theta", "theta omega"), ("omega^2", "omega^2"),
        ("theta", "theta"), ("omega", "omega"),
    ]
    for term in terms:
        term = term.strip()
        if not term:
            continue
        matched = False
        for pattern, coeff_key in term_patterns:
            if pattern in term:
                coeff_str = term.replace(pattern, "").replace("*", "").strip()
                try:
                    coeffs[coeff_key] = float(coeff_str) if coeff_str else 1.0
                except ValueError:
                    pass
                matched = True
                break
        if not matched:
            try:
                coeffs["1"] = float(term)
            except ValueError:
                pass
    return coeffs


def compute_scaling_params(theta, omega, t_scale=T_SCALE):
    import torch
    train_x = torch.tensor(np.stack([theta, omega], axis=1), dtype=torch.float32)
    mean_x = train_x.mean(dim=0).numpy()
    std_x = train_x.std(dim=0).numpy()
    std_x[std_x < 1e-6] = 1.0
    mean_x[1] = 0.0
    std_x[0] = std_x[1] * t_scale
    return mean_x, std_x


def simulate_ode_svise(t, theta0, omega0, coeffs_omega, mean_x, std_x, t_scale=T_SCALE):
    x0 = np.array([theta0, omega0])
    x0_scaled = (x0 - mean_x) / std_x
    t_scaled = t / t_scale

    def drift(state, t_):
        th, om = state
        domega = (coeffs_omega["1"]
                  + coeffs_omega["theta"] * th + coeffs_omega["omega"] * om
                  + coeffs_omega["theta^2"] * th**2 + coeffs_omega["theta omega"] * th * om
                  + coeffs_omega["omega^2"] * om**2
                  + coeffs_omega["theta^3"] * th**3 + coeffs_omega["theta^2 omega"] * th**2 * om
                  + coeffs_omega["theta omega^2"] * th * om**2 + coeffs_omega["omega^3"] * om**3)
        return [om, domega]

    sol_scaled = odeint(drift, x0_scaled, t_scaled, full_output=False)
    sol = sol_scaled * std_x + mean_x
    return sol[:, 0], sol[:, 1]


# ── SINDy equation parsing & simulation (from plot_forward_sim_sindy.py) ───────

def parse_equation_sindy(eq_str):
    coeffs = {
        "1": 0.0, "theta": 0.0, "omega": 0.0,
        "theta^2": 0.0, "theta omega": 0.0, "omega^2": 0.0,
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }
    eq_str = eq_str.replace("+ -", "+-").replace("- ", "-").replace("  ", " ").strip()
    terms = []
    current_term = ""
    for char in eq_str:
        if char == '+' and current_term.strip():
            terms.append(current_term.strip())
            current_term = ""
        else:
            current_term += char
    if current_term.strip():
        terms.append(current_term.strip())
    term_patterns = [
        ("theta^3", "theta^3"), ("theta^2 omega", "theta^2 omega"),
        ("theta omega^2", "theta omega^2"), ("omega^3", "omega^3"),
        ("theta^2", "theta^2"), ("theta omega", "theta omega"),
        ("omega theta", "theta omega"), ("omega^2", "omega^2"),
        ("theta", "theta"), ("omega", "omega"),
    ]
    for term in terms:
        term = term.strip()
        if not term:
            continue
        matched = False
        for pattern, coeff_key in term_patterns:
            if pattern in term:
                coeff_str = term.replace(pattern, "").replace("*", "").strip()
                try:
                    coeffs[coeff_key] = float(coeff_str) if coeff_str else 1.0
                except ValueError:
                    pass
                matched = True
                break
        if not matched:
            if term.endswith(" 1"):
                try:
                    coeffs["1"] = float(term[:-2].strip())
                except ValueError:
                    pass
            else:
                try:
                    coeffs["1"] = float(term)
                except ValueError:
                    pass
    return coeffs


def omega_coeffs_from_row(row):
    return {
        "1": float(row.get("Coeff_Const", 0) or 0),
        "theta": float(row.get("Coeff_Theta", 0) or 0),
        "omega": float(row.get("Coeff_Omega", 0) or 0),
        "theta^2": float(row.get("Coeff_Theta2", 0) or 0),
        "theta omega": float(row.get("Coeff_ThetaOmega", 0) or 0),
        "omega^2": float(row.get("Coeff_Omega2", 0) or 0),
        "theta^3": 0.0, "theta^2 omega": 0.0, "theta omega^2": 0.0, "omega^3": 0.0,
    }


def simulate_sindy_ode(t, theta0, omega0, coeffs_theta, coeffs_omega):
    tc = [coeffs_theta[k] for k in ["1", "theta", "omega", "theta^2", "theta omega", "omega^2"]]
    oc = [coeffs_omega[k] for k in ["1", "theta", "omega", "theta^2", "theta omega", "omega^2"]]

    def drift(state, t_):
        th, om = state
        dtheta = tc[0] + tc[1]*th + tc[2]*om + tc[3]*th*th + tc[4]*th*om + tc[5]*om*om
        domega = oc[0] + oc[1]*th + oc[2]*om + oc[3]*th*th + oc[4]*th*om + oc[5]*om*om
        return [dtheta, domega]

    sol = odeint(drift, np.array([theta0, omega0]), t, full_output=False)
    return sol[:, 0], sol[:, 1]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # --- Load empirical data ---
    print("Loading empirical data...")
    data = load_data(PARQUET_PATH)
    all_chunks = get_all_valid_chunks(data)
    chunk_start, chunk_df = all_chunks[CHUNK_IDX]
    print(f"Chunk {CHUNK_IDX}: {chunk_start}")

    freq_values = chunk_df['freq'].values
    if np.mean(freq_values) > 55:
        omega_raw = (freq_values - 60.0) * 2 * np.pi
    else:
        omega_raw = freq_values * 2 * np.pi

    omega_smooth = gaussian_filter1d(omega_raw, sigma=SIGMA)
    theta_raw = np.cumsum(omega_raw) * DT
    theta_smooth = np.cumsum(omega_smooth) * DT
    t = np.arange(300) * DT

    # --- SVISE forward simulation ---
    print("Running SVISE forward simulation...")
    df_svise = pd.read_csv(SVISE_CSV)
    row_svise = df_svise[df_svise["Chunk_Index"] == CHUNK_IDX].iloc[0]
    coeffs_svise = parse_equation_svise(str(row_svise["Eq_Omega"]))
    mean_x, std_x = compute_scaling_params(theta_raw, omega_raw)
    theta_sim_svise, omega_sim_svise = simulate_ode_svise(
        t, theta_raw[0], omega_raw[0], coeffs_svise, mean_x, std_x)

    diff_omega = float(row_svise.get("Diffusion_Omega", np.nan))
    has_tube = np.isfinite(diff_omega) and diff_omega > 0
    if has_tube:
        std_tube = 2.0 * np.sqrt(diff_omega * t)

    # --- SINDy forward simulation ---
    print("Running SINDy forward simulation...")
    df_sindy = pd.read_csv(SINDY_CSV)
    row_sindy = df_sindy[df_sindy["Chunk_Index"] == CHUNK_IDX].iloc[0]
    coeffs_theta_sindy = parse_equation_sindy(str(row_sindy["Eq_Theta"]))
    coeffs_omega_sindy = omega_coeffs_from_row(row_sindy)
    theta_sim_sindy, omega_sim_sindy = simulate_sindy_ode(
        t, theta_smooth[0], omega_smooth[0], coeffs_theta_sindy, coeffs_omega_sindy)

    # --- Figure ---
    print("Creating figure...")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.85,
        "xtick.major.width": 0.85,
        "ytick.major.width": 0.85,
    })

    c_raw = '#2196F3'
    c_smooth = '#4CAF50'
    c_sim = '#F44336'

    # Shared y-limits
    all_omega = np.concatenate([omega_raw, omega_smooth, omega_sim_sindy, omega_sim_svise])
    if has_tube:
        all_omega = np.concatenate([all_omega,
                                    omega_sim_svise + std_tube,
                                    omega_sim_svise - std_tube])
    ymin, ymax = np.nanmin(all_omega), np.nanmax(all_omega)
    ypad = 0.1 * (ymax - ymin)
    ylims = (ymin - ypad, ymax + ypad)

    fig = plt.figure(figsize=(10, 4.8))
    outer = fig.add_gridspec(2, 1, hspace=0.85, top=0.89, bottom=0.09,
                             left=0.06, right=0.98)

    # Row 1: [plot] [arrow] [plot] [arrow] [plot]  — wider arrow gaps
    gs1 = outer[0].subgridspec(1, 5, width_ratios=[3, 1.5, 3, 1.5, 3], wspace=0.05)
    ax1a = fig.add_subplot(gs1[0, 0])
    ax1b = fig.add_subplot(gs1[0, 2])
    ax1c = fig.add_subplot(gs1[0, 4])

    # Row 2: [plot] [------arrow------] [plot]
    gs2 = outer[1].subgridspec(1, 5, width_ratios=[3, 1.5, 3, 1.5, 3], wspace=0.05)
    ax2a = fig.add_subplot(gs2[0, 0])
    ax2b = fig.add_subplot(gs2[0, 4])

    # --- Row 1: SINDy / PySR path ---

    # (a) Raw empirical
    ax1a.plot(t, omega_raw, color=c_raw, linewidth=0.6, alpha=0.8)
    ax1a.set_title('(a) Empirical data', )
    ax1a.set_xlabel('Time (s)')
    ax1a.set_ylabel(r'$\omega$ (rad/s)')
    ax1a.set_xlim(0, 300)
    ax1a.set_ylim(ylims)

    # (b) Gaussian smoothed
    ax1b.plot(t, omega_smooth, color=c_smooth, linewidth=1.2)
    ax1b.set_title(r'(b) Smoothed ($\sigma$=15)', )
    ax1b.set_xlabel('Time (s)')
    ax1b.set_yticklabels([])
    ax1b.set_xlim(0, 300)
    ax1b.set_ylim(ylims)

    # (c) SINDy forward sim
    ax1c.plot(t, omega_sim_sindy, color=c_sim, linewidth=1.2, linestyle='--')
    ax1c.set_title('(c) Forward simulation', )
    ax1c.set_xlabel('Time (s)')
    ax1c.set_yticklabels([])
    ax1c.set_xlim(0, 300)
    ax1c.set_ylim(ylims)

    # --- Row 2: SVISE path ---

    # (d) Raw empirical
    ax2a.plot(t, omega_raw, color=c_raw, linewidth=0.6, alpha=0.8)
    ax2a.set_title('(d) Empirical data', )
    ax2a.set_xlabel('Time (s)')
    ax2a.set_ylabel(r'$\omega$ (rad/s)')
    ax2a.set_xlim(0, 300)
    ax2a.set_ylim(ylims)

    # (e) SVISE forward sim + diffusion band
    if has_tube:
        ax2b.fill_between(t, omega_sim_svise - std_tube, omega_sim_svise + std_tube,
                          color=c_sim, alpha=0.15, label=r'Diffusion $\pm 2\sigma$')
    ax2b.plot(t, omega_sim_svise, color=c_sim, linewidth=1.2, linestyle='--',
              label='Forward sim.')
    ax2b.set_title('(e) Forward simulation', )
    ax2b.set_xlabel('Time (s)')
    ax2b.set_yticklabels([])
    ax2b.set_xlim(0, 300)
    ax2b.set_ylim(ylims)
    ax2b.legend(loc='upper right', framealpha=0.8)

    # --- Arrows ---
    fig.canvas.draw()  # finalize layout before reading positions

    def add_arrow(fig, ax_left, ax_right, label, sublabel=None):
        bbox_l = ax_left.get_position()
        bbox_r = ax_right.get_position()
        y_mid = (bbox_l.y0 + bbox_l.y1) / 2
        x_start = bbox_l.x1 + 0.01
        x_end = bbox_r.x0 - 0.01
        x_mid = (x_start + x_end) / 2

        arrow = FancyArrowPatch(
            (x_start, y_mid), (x_end, y_mid),
            transform=fig.transFigure, clip_on=False,
            arrowstyle='->', mutation_scale=14,
            color='#333', linewidth=1.5,
        )
        fig.patches.append(arrow)
        fig.text(x_mid, y_mid + 0.02, label,
                 ha='center', va='bottom', fontsize=8, fontstyle='italic',
                 transform=fig.transFigure)
        if sublabel:
            fig.text(x_mid, y_mid - 0.02, sublabel,
                     ha='center', va='top', fontsize=7, color='#555',
                     transform=fig.transFigure)

    add_arrow(fig, ax1a, ax1b, 'Gaussian filtering', r'($\sigma$=15)')
    add_arrow(fig, ax1b, ax1c, 'SINDy / PySR')
    add_arrow(fig, ax2a, ax2b, 'SVISE', '(joint state estimation\n& equation discovery)')

    # --- Row labels (centered above each row) ---
    bbox1_l = ax1a.get_position()
    bbox1_r = ax1c.get_position()
    bbox2_l = ax2a.get_position()
    bbox2_r = ax2b.get_position()
    fig.text((bbox1_l.x0 + bbox1_r.x1) / 2, bbox1_l.y1 + 0.08,
             'Path A: Gaussian pre-filtering + sparse regression (SINDy, PySR)',
             ha='center', va='bottom', fontsize=9.5, fontweight='demibold',
             transform=fig.transFigure)
    fig.text((bbox2_l.x0 + bbox2_r.x1) / 2, bbox2_l.y1 + 0.08,
             'Path B: Variational inference on raw observations (SVISE)',
             ha='center', va='bottom', fontsize=9.5, fontweight='demibold',
             transform=fig.transFigure)

    # --- Save ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_png = os.path.join(OUTPUT_DIR, "preprocessing_paths_chunk93070.png")
    path_pdf = os.path.join(OUTPUT_DIR, "preprocessing_paths_chunk93070.pdf")
    fig.savefig(path_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path_png}")
    print(f"Saved: {path_pdf}")


if __name__ == "__main__":
    main()
