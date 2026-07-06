"""
1. Reads results_all_combined.csv
2. Expands each equation via sympy.expand()
3. Extracts coefficients robustly via sp.Poly
   → NaN when the term does NOT appear in the equation (not 0.0!)
4. Re-simulates each chunk and applies divergence filter (|omega_sim| > 0.4)
5. Computes RMSE and coefficient statistics for stable chunks only
"""

import os
import math
import pandas as pd
import numpy as np
import sympy as sp
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import odeint

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(SCRIPT_DIR, "results_all_combined.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "results_recomputed_coefs.csv")

# Data file
_DATA_BASE = os.path.join(SCRIPT_DIR, "..", "..", "dataset")
_PARQUET   = os.path.join(_DATA_BASE, "South_Korea_2024-08-15_2025-08-31_1s.parquet")
DATA_PATH  = _PARQUET

SIGMA      = 15
F_REF      = 60.0
DT         = 1.0
CHUNK_SIZE = 300

# Divergence threshold: if any |omega_sim| exceeds this, the chunk
# is considered divergent/unstable and excluded from RMSE statistics.
OMEGA_DIVERGENCE_THRESHOLD = 0.4
# ─────────────────────────────────────────────────────────────────────────────

# Symbols
x0, x1       = sp.symbols("x0 x1")
theta_sym, omega_sym = sp.symbols("theta omega")

COEF_MAP = {
    "const":             (0, 0),
    "theta_coef":        (1, 0),
    "omega_coef":        (0, 1),
    "omega_theta_coef":  (1, 1),
    "theta2_coef":       (2, 0),
    "omega2_coef":       (0, 2),
    "theta3_coef":       (3, 0),
    "theta2_omega_coef": (2, 1),
    "theta_omega2_coef": (1, 2),
    "omega3_coef":       (0, 3),
}

COEF_KEYS   = list(COEF_MAP.keys())
COEF_LABELS = ["1", "θ", "ω", "θω", "θ²", "ω²", "θ³", "θ²ω", "θω²", "ω³"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_path, limit_interpolation=10):
    print(f"Loading data from {data_path} ...")
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_pickle(data_path)

    if 'QI' in data.columns:
        data.loc[:, 'freq'] = data.loc[:, 'freq'].interpolate(
            method='time', limit=limit_interpolation)
        data.loc[data['freq'].isna(), 'QI'] = 2
        data.loc[~data['freq'].isna(), 'QI'] = 0
    else:
        data['freq'] = data['freq'].interpolate(
            method='time', limit=limit_interpolation)
    return data


def get_valid_chunks(data):
    print("Extracting valid 5-min chunks ...")
    if 'QI' in data.columns:
        data_filtered = data[(data['QI'] == 0) & data['freq'].notna()].dropna(subset=['freq'])
    else:
        data_filtered = data[data['freq'].notna()].dropna(subset=['freq'])

    chunk_groups = data_filtered.groupby(data_filtered.index.floor('5min'))
    valid_chunks = []
    for chunk_start, group in chunk_groups:
        if len(group) == CHUNK_SIZE:
            valid_chunks.append((chunk_start, group))

    print(f"Found {len(valid_chunks)} valid chunks.")
    return valid_chunks


def prepare_chunk(chunk_df, sigma=SIGMA):
    freq_values = chunk_df['freq'].values
    omega_raw = (freq_values - F_REF) * 2 * np.pi
    omega = gaussian_filter1d(omega_raw.astype(float), sigma=sigma) if sigma > 0 else omega_raw.astype(float)
    theta = np.cumsum(omega) * DT
    t = np.arange(len(omega)) * DT
    return t, theta, omega


# ── Equation parsing ──────────────────────────────────────────────────────────

def parse_equation(eq_str):
    """Parse PySR equation string, substitute x0->theta, x1->omega, expand."""
    if not isinstance(eq_str, str) or not eq_str.strip():
        return None
    try:
        expr = sp.sympify(eq_str)
        expr = expr.subs({x0: theta_sym, x1: omega_sym})
        expr = sp.expand(expr)
        return expr
    except Exception:
        return None


def extract_coefficients(expr):
    """
    Extract polynomial coefficients.
    Sets NaN when the term does not appear in the equation.
    sp.Poly.monoms() returns only the terms that are actually present —
    this distinguishes "term missing" (NaN) from "coefficient is 0".
    """
    result = {k: np.nan for k in COEF_KEYS}
    if expr is None:
        return result
    try:
        poly = sp.Poly(expr, theta_sym, omega_sym)
        present_monoms = set(poly.monoms())

        for name, (pt, po) in COEF_MAP.items():
            if (pt, po) in present_monoms:
                val = float(poly.coeff_monomial(theta_sym**pt * omega_sym**po))
                result[name] = val
    except Exception:
        pass
    return result


# ── ODE simulation ────────────────────────────────────────────────────────────

def make_full_poly_rhs(coeffs_dict):
    """Build RHS from coefficient dict (NaN treated as 0)."""
    c = {k: (v if np.isfinite(v) else 0.0) for k, v in coeffs_dict.items()}
    def rhs(t, y):
        th, om = y
        dw = (c["const"]
              + c["theta_coef"] * th
              + c["omega_coef"] * om
              + c["omega_theta_coef"] * om * th
              + c["theta2_coef"] * th**2
              + c["omega2_coef"] * om**2
              + c["theta3_coef"] * th**3
              + c["theta2_omega_coef"] * th**2 * om
              + c["theta_omega2_coef"] * th * om**2
              + c["omega3_coef"] * om**3)
        return [om, dw if math.isfinite(dw) else 0.0]
    return rhs


def simulate_chunk(t_arr, theta0, omega0, rhs_func):
    """Forward-simulate the ODE using odeint."""
    def rhs_odeint(y, t):
        return rhs_func(t, y)
    try:
        sol = odeint(rhs_odeint, [theta0, omega0], t_arr, full_output=False)
        if sol.shape[0] == len(t_arr):
            return sol[:, 0], sol[:, 1]
    except Exception:
        pass
    return np.full_like(t_arr, np.nan), np.full_like(t_arr, np.nan)


def compute_rmse(pred, true):
    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(val):
    if pd.isna(val):
        return f"{'NaN':>14}"
    if abs(val) < 0.001 or abs(val) > 10000:
        return f"{val:>14.3e}"
    return f"{val:>14.6f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {len(df):,} rows")

    # ── Deduplication ─────────────────────────────────────────────────────────
    df = df.sort_values("loss").drop_duplicates(subset="chunk_id", keep="first")
    print(f"After deduplication: {len(df):,} rows\n")

    # ── Recompute coefficients ────────────────────────────────────────────────
    print("Computing coefficients via sympy.expand()...")
    coef_rows = []
    errors = 0
    for i, eq in enumerate(df["equation"]):
        if i % 10000 == 0:
            print(f"  {i:,} / {len(df):,} processed...")
        expr = parse_equation(eq)
        coef_rows.append(extract_coefficients(expr))
        if expr is None:
            errors += 1

    coef_df = pd.DataFrame(coef_rows)

    # Overwrite old coefficient columns
    for col in COEF_KEYS:
        df[col] = coef_df[col].values

    df.to_csv(OUTPUT_CSV, index=False, na_rep="")
    print(f"\n✓ Saved: {OUTPUT_CSV}")
    print(f"  Parse errors: {errors:,}\n")

    # Plausibility check
    print("Plausibility check — chunks per term:")
    for key, label in zip(COEF_KEYS, COEF_LABELS):
        n = df[key].notna().sum()
        print(f"  {label:<8} : {n:>8,}  ({100*n/len(df):.1f}%)")

    # ── Load empirical data ───────────────────────────────────────────────────
    data = load_data(DATA_PATH)
    all_chunks = get_valid_chunks(data)
    chunk_lookup = {idx: (cs, cdf) for idx, (cs, cdf) in enumerate(all_chunks)}
    tstart_to_idx = {str(cs): idx for idx, (cs, _) in enumerate(all_chunks)}

    # ── Forward simulate with divergence filter ───────────────────────────────
    print(f"\nForward-simulating all chunks (divergence threshold: "
          f"|omega_sim| > {OMEGA_DIVERGENCE_THRESHOLD}) ...")

    df["sim_ok"] = df["sim_ok"].astype(str).str.strip().str.lower() == "true"
    ok_df = df[df["sim_ok"]].copy()

    rmse_values = []
    n_stable    = 0
    n_divergent = 0
    n_sim_fail  = 0

    for row_i, row in ok_df.iterrows():
        chunk_id = int(row["chunk_id"])
        t_start  = str(row["t_start"])

        if chunk_id not in chunk_lookup:
            if t_start in tstart_to_idx:
                chunk_id = tstart_to_idx[t_start]
            else:
                n_sim_fail += 1
                continue

        chunk_start, chunk_df = chunk_lookup[chunk_id]
        t_arr, theta, omega = prepare_chunk(chunk_df, sigma=SIGMA)

        # Build coefficients dict from this row
        coeffs_dict = {k: float(row[k]) if pd.notna(row[k]) else 0.0 for k in COEF_KEYS}

        rhs_func = make_full_poly_rhs(coeffs_dict)
        theta_sim, omega_sim = simulate_chunk(t_arr, theta[0], omega[0], rhs_func)

        # Solver failure
        if np.all(np.isnan(omega_sim)):
            n_sim_fail += 1
            continue

        # Divergence check
        max_abs_omega = np.nanmax(np.abs(omega_sim))
        if max_abs_omega > OMEGA_DIVERGENCE_THRESHOLD:
            n_divergent += 1
            continue

        # Stable
        rmse_omega = compute_rmse(omega_sim, omega)
        rmse_values.append(rmse_omega)
        n_stable += 1

        n_processed = n_stable + n_divergent + n_sim_fail
        if n_processed % 500 == 0:
            print(f"  Processed {n_processed} chunks "
                  f"({n_stable} stable, {n_divergent} divergent, "
                  f"{n_sim_fail} sim-fail) ...")

    n_ode_fail = (~df["sim_ok"]).sum()
    n_processed = n_stable + n_divergent + n_sim_fail
    rmse_arr = np.array(rmse_values)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"Chunk Status (divergence-filtered):")
    print(f"{'─'*75}")
    print(f"  Total chunks:                        {len(df):,}")
    print(f"  ODE solver failed (sim_ok=False):     {n_ode_fail:,}")
    print(f"  Evaluated (sim_ok=True):              {len(ok_df):,}")
    print(f"    Stable (|ω_sim| ≤ {OMEGA_DIVERGENCE_THRESHOLD}):            {n_stable:,}")
    print(f"    Divergent (|ω_sim| > {OMEGA_DIVERGENCE_THRESHOLD}):          {n_divergent:,}")
    print(f"    Sim-fail (re-sim):                  {n_sim_fail:,}")

    # ── RMSE Table ────────────────────────────────────────────────────────────
    if len(rmse_arr) > 0:
        print(f"\n{'─'*75}")
        print(f"Forward-Simulated RMSE (omega) — stable chunks only (n={n_stable:,})")
        print(f"{'─'*75}")
        print(f"  Mean:   {np.mean(rmse_arr):.6e}")
        print(f"  Std:    {np.std(rmse_arr):.6e}")
        print(f"  Median: {np.median(rmse_arr):.6e}")
        print(f"  Min:    {np.min(rmse_arr):.6e}")
        print(f"  Max:    {np.max(rmse_arr):.6e}")
        print(f"  25th %: {np.percentile(rmse_arr, 25):.6e}")
        print(f"  75th %: {np.percentile(rmse_arr, 75):.6e}")
    else:
        print("\n  No stable chunks found!")

    # ── Coefficient Table ─────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"Coefficient Statistics (chunks where term is present)")
    print(f"{'─'*75}")
    hdr2 = f"{'Term':<20} {'|Mean|':>14} {'Std':>14} {'n (non-NaN)':>16}"
    print(hdr2); print("─" * len(hdr2))

    for key, label in zip(COEF_KEYS, COEF_LABELS):
        col = df[key].replace([np.inf, -np.inf], np.nan).dropna()
        m = col.abs().mean() if len(col) > 0 else np.nan
        s = col.std()        if len(col) > 1 else np.nan
        n = len(col)
        print(f"{label:<20} {fmt(m)} {fmt(s)} {n:>16,}")


if __name__ == "__main__":
    main()

