import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# We import the exact data logic natively from run_analysis
from run_analysis_5min_all_chunks import load_data, get_all_valid_chunks, prepare_data

def unscale_equation(eq_str, mean_x, std_x, t_scale, feature_idx=1):
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    except ImportError:
        return "Install sympy: pip install sympy"

    if not isinstance(eq_str, str) or not eq_str or eq_str in ["N/A", "nan"] or "FAILED" in eq_str or "Error" in eq_str:
        return str(eq_str)

    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    try:
        expr = parse_expr(eq_str, transformations=transformations)
    except Exception as e:
        return f"Parse Error: {e}"

    # Target the arbitrary variable strings 
    x0, x1, theta, omega = sympy.symbols('x0 x1 theta omega')
    
    # Forward Scale Equation: x_scaled = (x_raw - mean) / std
    x0_sub = (x0 - mean_x[0]) / std_x[0]
    x1_sub = (x1 - mean_x[1]) / std_x[1]
    theta_sub = (theta - mean_x[0]) / std_x[0]
    omega_sub = (omega - mean_x[1]) / std_x[1]
    
    expr_sub = expr.subs({'x0': x0_sub, 'x1': x1_sub, 'theta': theta_sub, 'omega': omega_sub})
    
    # Complete the Derivative Unscaling: 
    # dx_phys / dt_phys = (std_x / t_scale) * expr_sub
    expr_phys = expr_sub * (std_x[feature_idx] / t_scale)
    expr_expanded = sympy.expand(expr_phys)
    
    for a in sympy.preorder_traversal(expr_expanded):
        if isinstance(a, sympy.Float):
            expr_expanded = expr_expanded.subs(a, round(a, 6))
            
    return str(expr_expanded)

def main():
    print("="*60)
    print(" 🛠️  RETROACTIVE EQUATION UNSCALER")
    print("="*60)
    
    # 1. Target aggregate CSV automatically based on recent directories or manual input
    target_path = input("Please paste the absolute path to your massive 'all_chunks_combined.csv':\n > ").strip()
    
    if not os.path.exists(target_path):
        print(f"\n[ERROR] CSV file not found at {target_path}")
        return
        
    df_results = pd.read_csv(target_path)
    print(f"\nLoaded {len(df_results):,} mathematical chunk limits from target CSV.")

    # 2. Re-load original Parquet dataset chunks to extract exactly perfect std/mean variances 
    data_path = os.path.join(os.path.dirname(__file__), "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    print(f"\nExtracing identical dataset constraints from {data_path} \nto dynamically reconstruct matrix normalizations...")
    
    data = load_data(data_path)
    all_chunks = get_all_valid_chunks(data)
    
    phys_eqs = []
    
    print("\nUnscaling abstract ML polynomials using sympy...")
    for idx_in_df, row in tqdm(df_results.iterrows(), total=len(df_results)):
        chunk_idx = int(row["Chunk_Index"])
        eq_om = str(row["Eq_Omega"])
        
        # Pluck exact mathematical chunk boundary matched by idx
        if chunk_idx < len(all_chunks):
            _, chunk_df = all_chunks[chunk_idx]
            
            # Reconstruct scaling statistics perfectly seamlessly using identical 0-sigma
            _, X_np, _ = prepare_data(chunk_df, dt=1.0, sigma=0)
            mean_x = np.mean(X_np, axis=0)
            std_x = np.std(X_np, axis=0)
            std_x[std_x < 1e-6] = 1.0
            t_scale = 30.0
            
            phys_eq = unscale_equation(eq_om, mean_x.tolist(), std_x.tolist(), t_scale, feature_idx=1)
            phys_eqs.append(phys_eq)
        else:
            phys_eqs.append("Out_Of_Bounds")
            
    df_results["Eq_Omega_Physical"] = phys_eqs
    
    # Write cleanly back to disk 
    output_file = target_path.replace(".csv", "_physical.csv")
    df_results.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully unscaled {len(df_results):,} dimensionless polynomial equations!")
    print(f"It was perfectly saved alongside the original run file as:\n  {output_file}")

if __name__ == '__main__':
    main()
