import csv
import os

csv_path = os.path.join(os.path.dirname(__file__), 'svise', 'results_5min_all_chunks', 'all_chunks_combined.csv')

best_rmse = float('inf')
best_chunk = None
best_time = None

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rmse = row['RMSE_Total'].strip()
        if rmse == '' or rmse == 'nan':
            continue
        rmse_val = float(rmse)
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_chunk = row['Chunk_Index']
            best_time = row['Chunk_Start_Time']

print(f'Best RMSE_Total: {best_rmse}')
print(f'Chunk Index: {best_chunk}')
print(f'Chunk Start Time: {best_time}')
