"""
Analyze dynamically active (sign-flip boundary) chunks from the 1h synthetic
noiseless SVISE validation.

The synthetic dataset uses dispatch=1 (Korea), where the perturbation sign
flips every 6 hours. With 1h chunks, boundary chunks have
Original_Chunk_Index % 6 == 0. These are the only chunks with sufficient
dynamical information for equation recovery.

This script:
  1. Loads the combined CSV from aggregate_results_synthetic_1h.py
  2. Identifies boundary chunks (Original_Chunk_Index % 6 == 0)
  3. Reports coefficient recovery statistics on ALL boundary chunks (no filtering)
  4. Additionally reports how many fall within a given error threshold
  5. Compares boundary vs non-boundary chunks

Usage:
    python analyze_boundary_chunks_1h.py --run-name run_SLURM_3730765
"""
import os
import re
import json
import argparse
import numpy as np
import pandas as pd


def parse_physical_equation(eq_str):
    """
    Parse a physical equation string like:
        '-0.008454*omega - 1.8e-5*theta + 0.005418'
    Returns dict with keys: c_omega, c_theta, intercept.
    """
    if not isinstance(eq_str, str) or eq_str.strip().lower() == "nan":
        return None

    c_omega = 0.0
    c_theta = 0.0

    m_omega = re.search(
        r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*omega', eq_str, re.IGNORECASE
    )
    if m_omega:
        c_omega = float(m_omega.group(1).replace(" ", ""))

    m_theta = re.search(
        r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)\s*\*\s*theta', eq_str, re.IGNORECASE
    )
    if m_theta:
        c_theta = float(m_theta.group(1).replace(" ", ""))

    # Intercept: standalone numbers not attached to *omega or *theta
    all_terms = re.findall(
        r'([+-]?\s*[\d.]+(?:e[+-]?\d+)?)(?:\s*\*\s*(omega|theta))?', eq_str
    )
    intercept = 0.0
    for val_str, var_name in all_terms:
        val_str = val_str.replace(" ", "")
        if not val_str:
            continue
        if var_name == "":
            try:
                intercept += float(val_str)
            except ValueError:
                pass

    return {"c_omega": c_omega, "c_theta": c_theta, "intercept": intercept}


def describe_coeff(series, gt_val, label):
    """Print and return stats for a coefficient series vs ground truth."""
    s = series.dropna()
    if len(s) == 0:
        print(f"  {label}: no valid values")
        return {}
    m, std = s.mean(), s.std()
    med = s.median()
    rel_errors = (s - gt_val).abs() / abs(gt_val)
    mean_rel = rel_errors.mean() * 100
    med_rel = rel_errors.median() * 100
    return {
        "mean": float(m), "std": float(std), "median": float(med),
        "mean_rel_error_pct": float(mean_rel),
        "median_rel_error_pct": float(med_rel),
        "count": int(len(s)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze boundary chunks from 1h synthetic validation"
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Folder name inside results_synthetic_noiseless_1h/"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(
        script_dir, "results_synthetic_noiseless_1h", args.run_name
    )
    csv_path = os.path.join(results_dir, "all_chunks_combined.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run aggregate_results_synthetic_1h.py first.")
        return

    # Load ground truth
    gt_path = os.path.join(script_dir, "ground_truth_params.json")
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    gt_c_omega = gt["c_1"]
    gt_c_theta = gt["c_2"]
    gt_delta_p = gt["Delta_P"]

    # Load data
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total chunks: {total}")

    # Identify valid (GP succeeded)
    numeric_cols = [
        "Orig_RMSE_Omega", "Sim_RMSE_Omega", "Sim_RMSE_Theta", "Sim_RMSE_Total"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[df["Orig_RMSE_Omega"].notna()].copy()
    failed = df[df["Orig_RMSE_Omega"].isna()]
    print(f"Succeeded: {len(valid)}, Failed: {len(failed)}")

    # Parse equations
    parsed = valid["Eq_Omega_Physical"].apply(parse_physical_equation)
    valid["c_omega"] = parsed.apply(lambda x: x["c_omega"] if x else np.nan)
    valid["c_theta"] = parsed.apply(lambda x: x["c_theta"] if x else np.nan)
    valid["intercept"] = parsed.apply(lambda x: x["intercept"] if x else np.nan)

    # =========================================================================
    # Boundary vs non-boundary split
    # =========================================================================
    valid["is_boundary"] = valid["Original_Chunk_Index"] % 6 == 0
    boundary_all = valid[valid["is_boundary"]].copy()
    non_boundary = valid[~valid["is_boundary"]].copy()

    # Count including failed chunks
    n_total_boundary = int((df["Original_Chunk_Index"] % 6 == 0).sum())
    n_total_non_boundary = int((df["Original_Chunk_Index"] % 6 != 0).sum())

    W = 75
    print(f"\n{'='*W}")
    print("CHUNK CATEGORIES")
    print(f"{'='*W}")
    print(f"Total boundary chunks (idx % 6 == 0):     {n_total_boundary}")
    print(f"Total non-boundary chunks:                 {n_total_non_boundary}")
    print(f"Boundary chunks with valid results:        {len(boundary_all)}")
    print(f"Non-boundary chunks with valid results:    {len(non_boundary)}")

    # =========================================================================
    # Compute relative errors on ALL boundary chunks (no filtering)
    # =========================================================================
    boundary_all["c_omega_rel_error"] = (
        (boundary_all["c_omega"] - gt_c_omega).abs() / abs(gt_c_omega)
    )
    boundary_all["c_theta_rel_error"] = (
        (boundary_all["c_theta"] - gt_c_theta).abs() / abs(gt_c_theta)
    )
    boundary_all["intercept_abs"] = boundary_all["intercept"].abs()
    boundary_all["intercept_rel_error"] = (
        (boundary_all["intercept_abs"] - gt_delta_p).abs() / abs(gt_delta_p)
    )

    # =========================================================================
    # Statistics on ALL boundary chunks (primary result - no cherry-picking)
    # =========================================================================
    n_boundary_valid = len(boundary_all)

    print(f"\n{'='*W}")
    print(f"COEFFICIENT RECOVERY ON ALL {n_boundary_valid} BOUNDARY CHUNKS")
    print(f"{'='*W}")
    print(f"{'Parameter':<25} {'Ground Truth':>15} {'Mean ± Std':>30} {'Med. Rel. Err':>12}")
    print(f"{'-'*W}")

    c_om = boundary_all["c_omega"]
    c_th = boundary_all["c_theta"]
    p0 = boundary_all["intercept_abs"]

    def fmt(series, gt_val):
        m, s = series.mean(), series.std()
        rel = float(((series - gt_val).abs() / abs(gt_val)).median()) * 100
        return f"{m:.5e} ± {s:.2e}", f"{rel:.1f}%"

    val_str, rel_str = fmt(c_om, gt_c_omega)
    print(f"{'c_omega':<25} {gt_c_omega:>15.5e} {val_str:>30} {rel_str:>12}")

    val_str, rel_str = fmt(c_th, gt_c_theta)
    print(f"{'c_theta':<25} {gt_c_theta:>15.5e} {val_str:>30} {rel_str:>12}")

    val_str, rel_str = fmt(p0, gt_delta_p)
    print(f"{'|P0| (intercept)':<25} {gt_delta_p:>15.5e} {val_str:>30} {rel_str:>12}")

    # RMSE on all boundary chunks
    sim_rmse = boundary_all["Sim_RMSE_Omega"].dropna()
    if len(sim_rmse) > 0:
        print(f"\n{'Sim RMSE omega':<25} {'---':>15} {sim_rmse.mean():.5e} ± {sim_rmse.std():.2e}")

    # =========================================================================
    # Non-boundary chunk stats for comparison
    # =========================================================================
    if len(non_boundary) > 0:
        non_boundary["c_omega_rel_error"] = (
            (non_boundary["c_omega"] - gt_c_omega).abs() / abs(gt_c_omega)
        )
        non_boundary["intercept_abs"] = non_boundary["intercept"].abs()

        print(f"\n{'='*W}")
        print(f"NON-BOUNDARY CHUNKS ({len(non_boundary)} valid) FOR COMPARISON")
        print(f"{'='*W}")
        nb_c_om = non_boundary["c_omega"]
        print(f"c_omega mean: {nb_c_om.mean():.5e} ± {nb_c_om.std():.2e}")
        print(f"c_omega median rel error: {non_boundary['c_omega_rel_error'].median()*100:.1f}%")
        nb_sim = non_boundary["Sim_RMSE_Omega"].dropna()
        if len(nb_sim) > 0:
            print(f"Sim RMSE omega: {nb_sim.mean():.5e} ± {nb_sim.std():.2e}")

    # =========================================================================
    # Error threshold breakdown (informational, not used for primary stats)
    # =========================================================================
    print(f"\n{'='*W}")
    print("ERROR THRESHOLD BREAKDOWN (c_omega relative error)")
    print(f"{'='*W}")
    for thresh in [0.05, 0.10, 0.20, 0.50, 1.00]:
        n_within = int((boundary_all["c_omega_rel_error"] < thresh).sum())
        print(f"  < {thresh*100:5.0f}%: {n_within:3d}/{n_boundary_valid} ({n_within/max(n_boundary_valid,1)*100:.1f}%)")

    # =========================================================================
    # LaTeX table values (copy-pasteable) — using ALL boundary chunks
    # =========================================================================
    print(f"\n{'='*W}")
    print("LATEX TABLE VALUES (all boundary chunks, no filtering)")
    print(f"{'='*W}")

    def latex_sci(series, exponent):
        """Format as (mantissa_mean +/- mantissa_std) x 10^exponent."""
        scale = 10**exponent
        m = series.mean() / scale
        s = series.std() / scale
        return f"({m:.2f} {{\\pm}} {s:.2f}){{\\times}}10^{{{exponent}}}"

    print(f"c_omega:    ${latex_sci(c_om, -3)}$")
    print(f"c_theta:    ${latex_sci(c_th, -5)}$")
    print(f"|P0|:       ${latex_sci(p0, -3)}$")
    if len(sim_rmse) > 0:
        print(f"RMSE omega: ${latex_sci(sim_rmse, -3)}$")

    # =========================================================================
    # Median relative errors (for text)
    # =========================================================================
    print(f"\n{'='*W}")
    print("MEDIAN RELATIVE ERRORS (all boundary chunks)")
    print(f"{'='*W}")
    med_c_omega = float(boundary_all["c_omega_rel_error"].median()) * 100
    med_c_theta = float(boundary_all["c_theta_rel_error"].median()) * 100
    med_p0 = float(boundary_all["intercept_rel_error"].median()) * 100
    print(f"c_omega:  {med_c_omega:.1f}%")
    print(f"c_theta:  {med_c_theta:.1f}%")
    print(f"|P0|:     {med_p0:.1f}%")

    # =========================================================================
    # Sign consistency of intercept (dispatch direction)
    # =========================================================================
    print(f"\n{'='*W}")
    print("INTERCEPT SIGN PATTERN (dispatch direction)")
    print(f"{'='*W}")
    n_pos = int((boundary_all["intercept"] > 0).sum())
    n_neg = int((boundary_all["intercept"] < 0).sum())
    print(f"Positive intercept: {n_pos}")
    print(f"Negative intercept: {n_neg}")
    print("(Sign should alternate across successive 6h boundaries)")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "run_name": args.run_name,
        "note": "Stats computed on ALL boundary chunks, no error-based filtering",
        "total_chunks": total,
        "total_boundary_chunks": n_total_boundary,
        "boundary_valid": n_boundary_valid,
        "ground_truth": {
            "c_omega": gt_c_omega,
            "c_theta": gt_c_theta,
            "Delta_P": gt_delta_p,
        },
        "all_boundary_stats": {
            "c_omega": describe_coeff(c_om, gt_c_omega, "c_omega"),
            "c_theta": describe_coeff(c_th, gt_c_theta, "c_theta"),
            "abs_intercept": describe_coeff(p0, gt_delta_p, "|P0|"),
            "sim_rmse_omega": {
                "mean": float(sim_rmse.mean()) if len(sim_rmse) > 0 else None,
                "std": float(sim_rmse.std()) if len(sim_rmse) > 0 else None,
                "count": int(len(sim_rmse)),
            },
        },
        "median_relative_errors_pct": {
            "c_omega": med_c_omega,
            "c_theta": med_c_theta,
            "abs_intercept": med_p0,
        },
        "error_threshold_breakdown": {
            f"<{int(t*100)}pct": int((boundary_all["c_omega_rel_error"] < t).sum())
            for t in [0.05, 0.10, 0.20, 0.50, 1.00]
        },
    }

    out_path = os.path.join(results_dir, "boundary_chunk_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nSaved: {out_path}")

    # Save all boundary chunk data (unfiltered)
    boundary_csv = os.path.join(results_dir, "boundary_chunks_all.csv")
    boundary_all.to_csv(boundary_csv, index=False)
    print(f"Saved: {boundary_csv}")


if __name__ == "__main__":
    main()
