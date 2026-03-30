"""
1. Liest results_all_combined.csv
2. Multipliziert jede equation mit sympy.expand() neu aus
3. Extrahiert Koeffizienten robust via sp.Poly
   → NaN wenn der Term NICHT in der Gleichung vorkommt (nicht 0.0!)
4. Berechnet RMSE- und Koeffizienten-Statistiken
"""

import pandas as pd
import numpy as np
import sympy as sp

# ── Konfiguration ─────────────────────────────────────────────────────────────
INPUT_CSV      = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/results_all_combined.csv"
OUTPUT_CSV     = "/home/ka/ka_iai/ka_hr7224/PySRCurrent/results_recomputed_coefs.csv"
RMSE_OMEGA_MAX = 1.0
RMSE_THETA_MAX = 10.0
# ─────────────────────────────────────────────────────────────────────────────

# Symbole
x0, x1       = sp.symbols("x0 x1")
theta, omega = sp.symbols("theta omega")

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


def parse_equation(eq_str):
    """Parse PySR equation string, substitute x0->theta, x1->omega, expand."""
    if not isinstance(eq_str, str) or not eq_str.strip():
        return None
    try:
        expr = sp.sympify(eq_str)
        expr = expr.subs({x0: theta, x1: omega})
        expr = sp.expand(expr)
        return expr
    except Exception:
        return None


def extract_coefficients(expr):
    """
    Extract polynomial coefficients.
    Setzt NaN wenn der Term nicht in der Gleichung vorkommt.
    sp.Poly.monoms() gibt nur die Terme zurück die wirklich vorkommen —
    damit unterscheiden wir "Term fehlt" (NaN) von "Koeffizient ist 0".
    """
    result = {k: np.nan for k in COEF_KEYS}
    if expr is None:
        return result
    try:
        poly = sp.Poly(expr, theta, omega)
        # Nur die Monome die wirklich in der Gleichung stehen
        present_monoms = set(poly.monoms())

        for name, (pt, po) in COEF_MAP.items():
            if (pt, po) in present_monoms:
                val = float(poly.coeff_monomial(theta**pt * omega**po))
                result[name] = val  # auch 0.0 eintragen falls explizit vorhanden
            # sonst bleibt NaN

    except Exception:
        pass
    return result


def fmt(val):
    if pd.isna(val):
        return f"{'NaN':>14}"
    if abs(val) < 0.001 or abs(val) > 10000:
        return f"{val:>14.3e}"
    return f"{val:>14.6f}"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Geladen: {len(df):,} Zeilen")

    # ── Deduplizierung ────────────────────────────────────────────────────────
    df = df.sort_values("loss").drop_duplicates(subset="chunk_id", keep="first")
    print(f"Nach Deduplizierung: {len(df):,} Zeilen\n")

    # ── Koeffizienten neu berechnen ───────────────────────────────────────────
    print("Berechne Koeffizienten via sympy.expand()...")
    coef_rows = []
    errors = 0
    for i, eq in enumerate(df["equation"]):
        if i % 10000 == 0:
            print(f"  {i:,} / {len(df):,} verarbeitet...")
        expr = parse_equation(eq)
        coef_rows.append(extract_coefficients(expr))
        if expr is None:
            errors += 1

    coef_df = pd.DataFrame(coef_rows)

    # Alte Koeffizienten-Spalten überschreiben
    for col in COEF_KEYS:
        df[col] = coef_df[col].values

    df.to_csv(OUTPUT_CSV, index=False, na_rep="")
    print(f"\n✓ Gespeichert: {OUTPUT_CSV}")
    print(f"  Fehler beim Parsen: {errors:,}\n")

    # Plausibilitätsprüfung
    print("Plausibilitätsprüfung — Chunks pro Term:")
    for key, label in zip(COEF_KEYS, COEF_LABELS):
        n = df[key].notna().sum()
        print(f"  {label:<8} : {n:>8,}  ({100*n/len(df):.1f}%)")

    # ── sim_ok Filter ─────────────────────────────────────────────────────────
    df["sim_ok"] = df["sim_ok"].astype(str).str.strip().str.lower() == "true"
    df["rmse_exploded"] = (
        (df["rmse_omega"] > RMSE_OMEGA_MAX) |
        (df["rmse_theta"] > RMSE_THETA_MAX) |
        df["rmse_omega"].isna() | df["rmse_theta"].isna()
    )
    df["successful"] = df["sim_ok"] & ~df["rmse_exploded"]
    ok = df[df["successful"]].copy()

    total = len(df); n_ok = len(ok); n_fail = total - n_ok
    print(f"\nChunk Status:")
    print(f"  Total      : {total:,}")
    print(f"  Successful : {n_ok:,}  ({100*n_ok/total:.1f}%)")
    print(f"  Failed     : {n_fail:,}  ({100*n_fail/total:.1f}%)")
    print(f"    davon ODE divergiert  : {(~df['sim_ok']).sum():,}")
    print(f"    davon RMSE explodiert : {(df['sim_ok'] & df['rmse_exploded']).sum():,}")

    # ── RMSE Tabelle ──────────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"RMSE Statistiken (nur successful chunks, n={n_ok:,})")
    print(f"{'─'*75}")
    hdr = f"{'Metric':<18} {'Mean':>14} {'Std':>14} {'Median':>14} {'Min':>14} {'Max':>14}"
    print(hdr); print("─" * len(hdr))

    for label, col in [("RMSE omega", "rmse_omega"), ("RMSE theta", "rmse_theta")]:
        vals = ok[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"{label:<18} {fmt(vals.mean())} {fmt(vals.std())} {fmt(vals.median())} {fmt(vals.min())} {fmt(vals.max())}")

    rmse_total = np.sqrt((ok["rmse_omega"]**2 + ok["rmse_theta"]**2) / 2).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"{'RMSE Total':<18} {fmt(rmse_total.mean())} {fmt(rmse_total.std())} {fmt(rmse_total.median())} {fmt(rmse_total.min())} {fmt(rmse_total.max())}")

    # ── Koeffizienten Tabelle ─────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"Koeffizienten Statistiken (nur Chunks wo Term wirklich vorkommt)")
    print(f"{'─'*75}")
    hdr2 = f"{'Term':<20} {'|Mean|':>14} {'Std':>14} {'n (nicht-NaN)':>16}"
    print(hdr2); print("─" * len(hdr2))

    for key, label in zip(COEF_KEYS, COEF_LABELS):
        col = df[key].replace([np.inf, -np.inf], np.nan).dropna()
        m = col.abs().mean() if len(col) > 0 else np.nan
        s = col.std()        if len(col) > 1 else np.nan
        n = len(col)
        print(f"{label:<20} {fmt(m)} {fmt(s)} {n:>16,}")


if __name__ == "__main__":
    main()
