#!/bin/bash
#SBATCH --job-name=svise_5min
#SBATCH --time=5:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-5min/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the project root directory
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

# Run the python script (5-minute chunks)
python -u run_analysis_5min_chunks.py --model integrator --sigma 60 --epochs 5000 --degree 2 --tau 1e-1 --chunk 50 --lr 0.001
