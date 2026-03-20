import os
import sys
import multiprocessing as mp
import time
from tqdm import tqdm

# Import your core logic directly to avoid code duplication
import run_analysis_5min_all_chunks as main_script

# Memory safeguard
def init_worker():
    import warnings
    warnings.filterwarnings('ignore')

def main():
    print("=" * 60)
    print(" LOCAL SVISE PARALLEL ACCELERATOR")
    print("=" * 60)
    
    print("\nLoading massive dataset ONCE in the main thread to protect your RAM...")
    data_path = os.path.join(os.path.dirname(__file__), "../dataset/South_Korea_2024-08-15_2025-08-31_1s.parquet")
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR] I couldn't find the dataset at:\n  {os.path.abspath(data_path)}")
        print("Please make sure you downloaded the dataset onto your computer first!")
        return

    data = main_script.load_data(data_path)
    all_chunks = main_script.get_all_valid_chunks(data)
    total_chunks = len(all_chunks)

    print(f"Extraction successful: Found {total_chunks:,} valid 5-minute chunks.")
    print("Deleting massive raw dataframe from RAM before spawning workers to prevent Out-Of-Memory crashes...")
    del data # Frees 4GB of RAM immediately!
    
    # We strip the tuple down to just the DataFrame dictionary payload to reduce parallel transit latency
    chunk_dfs = [chunk[1] for chunk in all_chunks]
    
    # We politely leave 2 CPU cores free so your PC stays responsive to mouse clicks and Chrome
    MAX_CORES = max(1, mp.cpu_count() - 2)
    print(f"\nSpawning isolated Python sub-processes dynamically across {MAX_CORES} physical CPU threads...")
    
    results = []
    start_time = time.time()
    
    # Using mp.Pool securely handles Windows and Linux natively on consumer architectures
    with mp.Pool(processes=MAX_CORES, initializer=init_worker) as pool:
        # We hook into tqdm to give you a gorgeous live CLI progress bar on your terminal!
        future_results = pool.imap(main_script.train_single_chunk, chunk_dfs)
        for res in tqdm(future_results, total=total_chunks, desc="Evaluating chunks", unit="chunk"):
            results.append(res)

    end_time = time.time()
    print(f"\nFinished evaluating all {total_chunks:,} chunks seamlessly in {(end_time-start_time)/60:.1f} minutes!")

    # Instantly aggregate results into one perfect file (no SLURM messy folder needed)
    import csv
    results_dir = os.path.join(os.path.dirname(__file__), "results_5min_all_chunks", "run_LOCAL_DESKTOP")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "chunks_all_combined_local.csv")
    
    print(f"Generating absolute single master spreadsheet securely at:\n  {csv_path}")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Chunk_Index", "Chunk_Start_Time",
            "RMSE_Omega", "RMSE_Theta", "RMSE_Total",
            "Final_Loss", "Stopped_Epoch", "NaN_Recoveries",
            "Eq_Theta", "Eq_Omega"
        ])
        
        for i, (chunk_start_time, _) in enumerate(all_chunks):
            res_dict = results[i]
            eq_theta = res_dict["equations"][0] if len(res_dict["equations"]) > 0 else "N/A"
            eq_omega = res_dict["equations"][1] if len(res_dict["equations"]) > 1 else "N/A"

            try:
                rmse_om = f"{float(res_dict['rmse_omega']):.6f}"
                rmse_th = f"{float(res_dict['rmse_theta']):.6f}"
                rmse_tot = f"{float(res_dict['rmse_total']):.6f}"
            except:
                rmse_om, rmse_th, rmse_tot = "nan", "nan", "nan"

            writer.writerow([
                i, str(chunk_start_time),
                rmse_om, rmse_th, rmse_tot,
                res_dict.get('final_loss', 'nan'), res_dict.get('stopped_epoch', -1), res_dict.get('nan_recoveries', 0),
                eq_theta, eq_omega
            ])
            
    print("\nCompletely finished! You are good to close the terminal.")

if __name__ == '__main__':
    main()
