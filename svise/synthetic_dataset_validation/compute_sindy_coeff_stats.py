"""Quick script to compute mean/std of SINDy coefficients and compare to ground truth.

Focuses on boundary chunks (Original_Chunk_Index % 6 == 0) which are the only
chunks with sufficient dynamics for equation recovery (sign-flip boundaries).
"""
import pandas as pd
import numpy as np
import json
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# Find the CSV
results_dir = os.path.join(script_dir, "results_sindy_synthetic_noiseless_1h")
# Find the most recent run
runs = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])
if not runs:
    print("No runs found!")
    sys.exit(1)

run_dir = os.path.join(results_dir, runs[-1])
csv_files = [f for f in os.listdir(run_dir) if f.endswith('.csv')]
if not csv_files:
    print("No CSV files found!")
    sys.exit(1)

csv_path = os.path.join(run_dir, csv_files[0])
print(f"Reading: {csv_path}\n")

df_all = pd.read_csv(csv_path)

# Filter to boundary chunks only
df = df_all[df_all["Original_Chunk_Index"] % 6 == 0].copy()
print(f"Total chunks: {len(df_all)}")
print(f"Boundary chunks (orig_idx % 6 == 0): {len(df)}")
print()

coeff_cols = [c for c in df.columns if c.startswith("Coeff_")]

print("=" * 65)
print("SINDy Coefficient Statistics — BOUNDARY CHUNKS ONLY (mean ± std)")
print("=" * 65)
for col in coeff_cols:
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    print(f"  {col:25s}: {vals.mean():+.10e} ± {vals.std():.10e}")

# Ground truth comparison
gt_path = os.path.join(script_dir, "ground_truth_params.json")
with open(gt_path) as f:
    gt = json.load(f)

c1 = gt["c_1"]
c2_decay = gt["c_2_decay"]
c2 = c2_decay * c1  # effective theta coefficient
delta_p = gt["Delta_P"]

print()
print("=" * 65)
print("Ground Truth (from generate_synthetic_data.py)")
print("=" * 65)
print(f"  Equation:  domega/dt = c_1*omega + c_2_decay*c_1*theta + Delta_P*P(t)")
print(f"  c_1             = {c1:+.10e}   (maps to Coeff_Omega)")
print(f"  c_2_decay * c_1 = {c2:+.10e}   (maps to Coeff_Theta)")
print(f"  Delta_P         = {delta_p:+.10e}   (time-varying forcing, ~Coeff_Const avg)")
print(f"  All other coefficients should be ~0")
print()

# Direct comparison — boundary chunks
print("=" * 65)
print("Comparison: SINDy mean vs Ground Truth (BOUNDARY CHUNKS)")
print("=" * 65)
sindy_const = pd.to_numeric(df["Coeff_Const"], errors="coerce").dropna()
sindy_theta = pd.to_numeric(df["Coeff_Theta"], errors="coerce").dropna()
sindy_omega = pd.to_numeric(df["Coeff_Omega"], errors="coerce").dropna()

print(f"  {'Term':20s} {'SINDy Mean':>18s}  {'Ground Truth':>18s}  {'Abs Error':>14s}")
print(f"  {'-'*20} {'-'*18}  {'-'*18}  {'-'*14}")
print(f"  {'Const (Delta_P)':20s} {sindy_const.mean():+.10e}  {delta_p:+.10e}  {abs(sindy_const.mean()-delta_p):.4e}")
print(f"  {'Theta (c2*c1)':20s} {sindy_theta.mean():+.10e}  {c2:+.10e}  {abs(sindy_theta.mean()-c2):.4e}")
print(f"  {'Omega (c1)':20s} {sindy_omega.mean():+.10e}  {c1:+.10e}  {abs(sindy_omega.mean()-c1):.4e}")

# Nonlinear terms (only present if degree > 1)
for col in ["Coeff_Theta2", "Coeff_ThetaOmega", "Coeff_Omega2"]:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        print(f"  {col.replace('Coeff_',''):20s} {vals.mean():+.10e}  {0.0:+.10e}  {abs(vals.mean()):.4e}")

# RMSE and convergence stats for boundary chunks
print()
print("=" * 65)
print("Simulation RMSE & Convergence — BOUNDARY CHUNKS")
print("=" * 65)
sim_rmse = pd.to_numeric(df["Sim_RMSE_Omega"], errors="coerce")
n_converged = sim_rmse.notna().sum()
n_total_boundary = len(df)
valid_rmse = sim_rmse.dropna()
print(f"  Boundary chunks total:     {n_total_boundary}")
print(f"  Converged (valid sim):     {n_converged}/{n_total_boundary} ({100*n_converged/max(n_total_boundary,1):.1f}%)")
if len(valid_rmse) > 0:
    print(f"  Sim RMSE omega mean:       {valid_rmse.mean():.6e}")
    print(f"  Sim RMSE omega std:        {valid_rmse.std():.6e}")
    print(f"  Sim RMSE omega median:     {valid_rmse.median():.6e}")
    print(f"  Sim RMSE omega min/max:    {valid_rmse.min():.6e} / {valid_rmse.max():.6e}")

sim_rmse_theta = pd.to_numeric(df["Sim_RMSE_Theta"], errors="coerce").dropna()
sim_rmse_total = pd.to_numeric(df["Sim_RMSE_Total"], errors="coerce").dropna()
if len(sim_rmse_theta) > 0:
    print(f"  Sim RMSE theta mean:       {sim_rmse_theta.mean():.6e}")
    print(f"  Sim RMSE theta std:        {sim_rmse_theta.std():.6e}")
if len(sim_rmse_total) > 0:
    print(f"  Sim RMSE total mean:       {sim_rmse_total.mean():.6e}")
    print(f"  Sim RMSE total std:        {sim_rmse_total.std():.6e}")

# Also show all-chunks stats for comparison
print()
print("=" * 65)
print("For reference: ALL chunks (including non-boundary)")
print("=" * 65)
all_const = pd.to_numeric(df_all["Coeff_Const"], errors="coerce").dropna()
all_theta = pd.to_numeric(df_all["Coeff_Theta"], errors="coerce").dropna()
all_omega = pd.to_numeric(df_all["Coeff_Omega"], errors="coerce").dropna()
print(f"  {'Const (Delta_P)':20s} {all_const.mean():+.10e}  (n={len(all_const)})")
print(f"  {'Theta (c2*c1)':20s} {all_theta.mean():+.10e}  (n={len(all_theta)})")
print(f"  {'Omega (c1)':20s} {all_omega.mean():+.10e}  (n={len(all_omega)})")
