"""
Analyze PySR coefficient recovery on dynamically active (sign-flip boundary)
chunks from the 60-min synthetic noiseless dataset.

Boundary chunks: chunk_id % 6 == 0 (sign flips every 6 hours).

Reads all_results_60min_combined.csv (or recomputed_coefs variant),
computes recovery statistics on boundary chunks vs ground truth,
and outputs LaTeX-ready values matching the SVISE analysis format.

Usage:
    python analyze_boundary_chunks_60min.py
    python analyze_boundary_chunks_60min.py --csv all_results_60min_recomputed_coefs.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
import json


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PySR boundary chunk recovery on 60min synthetic data"
    )
    parser.add_argument(
        "--csv", type=str, default="all_results_60min_combined.csv",
        help="CSV file name (default: all_results_60min_combined.csv)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.csv)

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        return

    # Ground truth (same as SVISE synthetic validation)
    gt_c_omega = -0.009057647473133581
    gt_c_theta = -1.5317625512028024e-05
    gt_delta_p = 0.00554890484189234

    # Load data
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Total chunks: {total}")
    print(f"Columns: {list(df.columns)}")

    # Identify coefficient columns
    # PySR CSV has: const, omega_coef, theta_coef, etc.
    coef_cols = ["const", "omega_coef", "theta_coef"]
    for col in coef_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Also ensure rmse columns are numeric
    for col in ["rmse_omega", "rmse_omega_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # =========================================================================
    # Boundary vs non-boundary split
    # =========================================================================
    df["is_boundary"] = df["chunk_id"] % 6 == 0
    boundary = df[df["is_boundary"]].copy()
    non_boundary = df[~df["is_boundary"]].copy()

    n_total_boundary = int(df["is_boundary"].sum())
    n_total_non_boundary = total - n_total_boundary

    W = 75
    print(f"\n{'='*W}")
    print("CHUNK CATEGORIES")
    print(f"{'='*W}")
    print(f"Total boundary chunks (chunk_id % 6 == 0):  {n_total_boundary}")
    print(f"Total non-boundary chunks:                   {n_total_non_boundary}")

    # =========================================================================
    # Map PySR coefficients to SVISE naming
    # c_omega = omega_coef (coefficient of omega in domega/dt)
    # c_theta = theta_coef
    # |P0|    = |const| (intercept)
    # =========================================================================
    boundary["c_omega"] = boundary["omega_coef"].fillna(0.0)
    boundary["c_theta"] = boundary["theta_coef"].fillna(0.0)
    boundary["intercept"] = boundary["const"].fillna(0.0)
    boundary["intercept_abs"] = boundary["intercept"].abs()

    # Relative errors
    boundary["c_omega_rel_error"] = (
        (boundary["c_omega"] - gt_c_omega).abs() / abs(gt_c_omega)
    )
    boundary["c_theta_rel_error"] = (
        (boundary["c_theta"] - gt_c_theta).abs() / abs(gt_c_theta)
    )
    boundary["intercept_rel_error"] = (
        (boundary["intercept_abs"] - gt_delta_p).abs() / abs(gt_delta_p)
    )

    n_boundary = len(boundary)

    # =========================================================================
    # Statistics on ALL boundary chunks
    # =========================================================================
    c_om = boundary["c_omega"]
    c_th = boundary["c_theta"]
    p0 = boundary["intercept_abs"]

    print(f"\n{'='*W}")
    print(f"COEFFICIENT RECOVERY ON ALL {n_boundary} BOUNDARY CHUNKS")
    print(f"{'='*W}")
    print(f"{'Parameter':<25} {'Ground Truth':>15} {'Mean ± Std':>30} {'Med. Rel. Err':>12}")
    print(f"{'-'*W}")

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

    # RMSE
    sim_rmse = boundary["rmse_omega"].dropna()
    if len(sim_rmse) > 0:
        print(f"\n{'Sim RMSE omega':<25} {'---':>15} {sim_rmse.mean():.5e} ± {sim_rmse.std():.2e}")

    # =========================================================================
    # Non-boundary for comparison
    # =========================================================================
    if len(non_boundary) > 0:
        non_boundary["c_omega"] = non_boundary["omega_coef"].fillna(0.0)
        nb_rel = ((non_boundary["c_omega"] - gt_c_omega).abs() / abs(gt_c_omega))

        print(f"\n{'='*W}")
        print(f"NON-BOUNDARY CHUNKS ({len(non_boundary)}) FOR COMPARISON")
        print(f"{'='*W}")
        print(f"c_omega mean: {non_boundary['c_omega'].mean():.5e} ± {non_boundary['c_omega'].std():.2e}")
        print(f"c_omega median rel error: {nb_rel.median()*100:.1f}%")
        nb_sim = non_boundary["rmse_omega"].dropna()
        if len(nb_sim) > 0:
            print(f"Sim RMSE omega: {nb_sim.mean():.5e} ± {nb_sim.std():.2e}")

    # =========================================================================
    # Error threshold breakdown
    # =========================================================================
    print(f"\n{'='*W}")
    print("ERROR THRESHOLD BREAKDOWN (c_omega relative error)")
    print(f"{'='*W}")
    for thresh in [0.05, 0.10, 0.20, 0.50, 1.00]:
        n_within = int((boundary["c_omega_rel_error"] < thresh).sum())
        print(f"  < {thresh*100:5.0f}%: {n_within:3d}/{n_boundary} ({n_within/max(n_boundary,1)*100:.1f}%)")

    # =========================================================================
    # LaTeX table values
    # =========================================================================
    print(f"\n{'='*W}")
    print("LATEX TABLE VALUES (all boundary chunks)")
    print(f"{'='*W}")

    def latex_sci(series, exponent):
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
    # Median relative errors
    # =========================================================================
    print(f"\n{'='*W}")
    print("MEDIAN RELATIVE ERRORS (all boundary chunks)")
    print(f"{'='*W}")
    med_c_omega = float(boundary["c_omega_rel_error"].median()) * 100
    med_c_theta = float(boundary["c_theta_rel_error"].median()) * 100
    med_p0 = float(boundary["intercept_rel_error"].median()) * 100
    print(f"c_omega:  {med_c_omega:.1f}%")
    print(f"c_theta:  {med_c_theta:.1f}%")
    print(f"|P0|:     {med_p0:.1f}%")

    # =========================================================================
    # Intercept sign pattern
    # =========================================================================
    print(f"\n{'='*W}")
    print("INTERCEPT SIGN PATTERN (dispatch direction)")
    print(f"{'='*W}")
    n_pos = int((boundary["intercept"] > 0).sum())
    n_neg = int((boundary["intercept"] < 0).sum())
    n_zero = int((boundary["intercept"] == 0).sum())
    print(f"Positive intercept: {n_pos}")
    print(f"Negative intercept: {n_neg}")
    print(f"Zero intercept:     {n_zero}")

    # =========================================================================
    # Sample equations from boundary chunks
    # =========================================================================
    print(f"\n{'='*W}")
    print("SAMPLE EQUATIONS (first 10 boundary chunks)")
    print(f"{'='*W}")
    for _, row in boundary.head(10).iterrows():
        cid = int(row["chunk_id"])
        eq = row.get("equation", "N/A")
        rmse = row.get("rmse_omega", float("nan"))
        rmse_str = f"{rmse:.6f}" if pd.notna(rmse) else "N/A"
        print(f"  Chunk {cid:4d}: {eq}  [RMSE: {rmse_str}]")

    # =========================================================================
    # Save results
    # =========================================================================
    output = {
        "csv_file": args.csv,
        "note": "Stats computed on ALL boundary chunks, no error-based filtering",
        "total_chunks": total,
        "total_boundary_chunks": n_total_boundary,
        "ground_truth": {
            "c_omega": gt_c_omega,
            "c_theta": gt_c_theta,
            "Delta_P": gt_delta_p,
        },
        "all_boundary_stats": {
            "c_omega": {
                "mean": float(c_om.mean()), "std": float(c_om.std()),
                "median": float(c_om.median()), "count": int(len(c_om)),
                "median_rel_error_pct": med_c_omega,
            },
            "c_theta": {
                "mean": float(c_th.mean()), "std": float(c_th.std()),
                "median": float(c_th.median()), "count": int(len(c_th)),
                "median_rel_error_pct": med_c_theta,
            },
            "abs_intercept": {
                "mean": float(p0.mean()), "std": float(p0.std()),
                "median": float(p0.median()), "count": int(len(p0)),
                "median_rel_error_pct": med_p0,
            },
            "sim_rmse_omega": {
                "mean": float(sim_rmse.mean()) if len(sim_rmse) > 0 else None,
                "std": float(sim_rmse.std()) if len(sim_rmse) > 0 else None,
                "count": int(len(sim_rmse)),
            },
        },
        "error_threshold_breakdown": {
            f"<{int(t*100)}pct": int((boundary["c_omega_rel_error"] < t).sum())
            for t in [0.05, 0.10, 0.20, 0.50, 1.00]
        },
    }

    out_path = os.path.join(script_dir, "boundary_chunk_analysis_60min.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nSaved: {out_path}")

    boundary_csv = os.path.join(script_dir, "boundary_chunks_60min_all.csv")
    boundary.to_csv(boundary_csv, index=False)
    print(f"Saved: {boundary_csv}")


if __name__ == "__main__":
    main()
