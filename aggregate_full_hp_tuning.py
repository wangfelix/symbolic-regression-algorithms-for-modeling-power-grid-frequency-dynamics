"""
Aggregate HP Tuning Results for SINDy and SVISE (Full Synthetic Dataset)

Usage:
    python aggregate_full_hp_tuning.py --model sindy --csv
    python aggregate_full_hp_tuning.py --model svise --csv
"""
import os, sys, json, glob, argparse
import numpy as np

def auto_detect_run_dir(model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if model == "sindy":
        base = os.path.join(script_dir, "SINDy", "results_sindy_synthetic_full_hp_tuning")
    else:
        base = os.path.join(script_dir, "SVISE", "synthetic_dataset_validation",
                            "results_svise_synthetic_full_hp_tuning")
    if not os.path.isdir(base):
        print(f"Not found: {base}"); return None
    runs = sorted(os.listdir(base))
    return os.path.join(base, runs[-1]) if runs else None

def load_results(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "combo_*.json")))
    return [json.load(open(f)) for f in files]

def print_gt():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "SVISE", "synthetic_dataset_validation", "ground_truth_params.json")
    if os.path.exists(p):
        gt = json.load(open(p))
        print(f"\nGround Truth:  c_1={gt.get('c_1')}  c_2={gt.get('c_2')}  Delta_P={gt.get('Delta_P')}")

def sort_key(r):
    v = r.get("sim_rmse_omega", float('nan'))
    return v if np.isfinite(v) else 1e30

def print_sindy(results, top=20):
    rs = sorted(results, key=sort_key)
    print(f"\n{'Rank':>4} {'Idx':>3} {'sig':>4} {'deg':>3} {'threshold':>10} "
          f"{'RMSE_w':>12} {'Status':>12} {'Const':>12} {'Theta':>12} {'Omega':>12} "
          f"{'err_c1%':>8} {'err_c2%':>8} {'err_DP%':>8}")
    print("-"*130)
    for i, r in enumerate(rs[:top], 1):
        hp, c, gt = r.get("hyperparams",{}), r.get("coefficients",{}), r.get("ground_truth_comparison",{})
        print(f"{i:4d} {r.get('combo_index','?'):>3} {hp.get('sigma','?'):>4} {hp.get('degree','?'):>3} "
              f"{hp.get('threshold','?'):>10.0e} {r.get('sim_rmse_omega',float('nan')):>12.6e} "
              f"{r.get('sim_status','?'):>12} "
              f"{c.get('Coeff_Const',0):>+12.4e} {c.get('Coeff_Theta',0):>+12.4e} {c.get('Coeff_Omega',0):>+12.4e} "
              f"{gt.get('Coeff_Omega',{}).get('rel_error_pct',float('nan')):>8.1f} "
              f"{gt.get('Coeff_Theta',{}).get('rel_error_pct',float('nan')):>8.1f} "
              f"{gt.get('Coeff_Const',{}).get('rel_error_pct',float('nan')):>8.1f}")
    if len(rs) > top: print(f"  ... ({len(rs)-top} more)")

def print_svise(results, top=20):
    rs = sorted(results, key=sort_key)
    print(f"\n{'Rank':>4} {'Idx':>3} {'deg':>3} {'tau':>8} {'lr':>8} {'n_tau':>5} "
          f"{'noise':>8} {'n_rep':>5} {'Loss':>10} {'RMSE_w':>12} {'Status':>12} "
          f"{'Const':>12} {'Theta':>12} {'Omega':>12} "
          f"{'err_c1%':>8} {'err_c2%':>8} {'err_DP%':>8}")
    print("-"*170)
    for i, r in enumerate(rs[:top], 1):
        hp, c, gt = r.get("hyperparams",{}), r.get("coefficients",{}), r.get("ground_truth_comparison",{})
        tr = r.get("training",{})
        print(f"{i:4d} {r.get('combo_index','?'):>3} {hp.get('degree','?'):>3} {hp.get('tau','?'):>8.0e} "
              f"{hp.get('lr','?'):>8.0e} {hp.get('n_tau','?'):>5} {hp.get('measurement_noise','?'):>8.0e} "
              f"{hp.get('n_reparam_samples','?'):>5} {tr.get('final_loss',float('nan')):>10.1f} "
              f"{r.get('sim_rmse_omega',float('nan')):>12.6e} {r.get('sim_status','?'):>12} "
              f"{c.get('Coeff_Const',0):>+12.4e} {c.get('Coeff_Theta',0):>+12.4e} {c.get('Coeff_Omega',0):>+12.4e} "
              f"{gt.get('Coeff_Omega',{}).get('rel_error_pct',float('nan')):>8.1f} "
              f"{gt.get('Coeff_Theta',{}).get('rel_error_pct',float('nan')):>8.1f} "
              f"{gt.get('Coeff_Const',{}).get('rel_error_pct',float('nan')):>8.1f}")
    if len(rs) > top: print(f"  ... ({len(rs)-top} more)")

def save_csv(results, path, model):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        if model == "sindy":
            w.writerow(["combo","sigma","degree","threshold","sim_rmse_omega","status",
                        "Const","Theta","Omega","err_c1%","err_c2%","err_dp%","equation"])
        else:
            w.writerow(["combo","degree","tau","lr","n_tau","noise","n_rep",
                        "loss","best_epoch","sim_rmse_omega","gp_rmse_omega","status",
                        "Const","Theta","Omega","err_c1%","err_c2%","err_dp%","equation"])
        for r in sorted(results, key=sort_key):
            hp,c,gt = r.get("hyperparams",{}),r.get("coefficients",{}),r.get("ground_truth_comparison",{})
            errs = [gt.get(k,{}).get("rel_error_pct","") for k in ["Coeff_Omega","Coeff_Theta","Coeff_Const"]]
            if model == "sindy":
                w.writerow([r.get("combo_index"),hp.get("sigma"),hp.get("degree"),hp.get("threshold"),
                           r.get("sim_rmse_omega"),r.get("sim_status"),
                           c.get("Coeff_Const",0),c.get("Coeff_Theta",0),c.get("Coeff_Omega",0),
                           *errs, r.get("equations",{}).get("d_omega_dt","")])
            else:
                tr = r.get("training",{})
                w.writerow([r.get("combo_index"),hp.get("degree"),hp.get("tau"),hp.get("lr"),
                           hp.get("n_tau"),hp.get("measurement_noise"),hp.get("n_reparam_samples"),
                           tr.get("final_loss"),tr.get("best_epoch"),
                           r.get("sim_rmse_omega"),r.get("gp_rmse_omega"),r.get("sim_status"),
                           c.get("Coeff_Const",0),c.get("Coeff_Theta",0),c.get("Coeff_Omega",0),
                           *errs, r.get("equations",{}).get("physical_omega","")])
    print(f"\nSaved: {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["sindy","svise"], required=True)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--csv", action="store_true")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()
    run_dir = args.run_dir or auto_detect_run_dir(args.model)
    if not run_dir or not os.path.isdir(run_dir):
        print(f"Not found: {run_dir}"); return
    results = load_results(run_dir)
    print(f"Loaded {len(results)} results from {run_dir}")
    print_gt()
    (print_sindy if args.model == "sindy" else print_svise)(results, args.top)
    if args.csv:
        save_csv(results, os.path.join(run_dir, f"summary_{args.model}.csv"), args.model)

if __name__ == "__main__":
    main()
