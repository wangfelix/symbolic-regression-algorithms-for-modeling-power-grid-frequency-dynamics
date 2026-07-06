#!/bin/bash
#SBATCH --job-name=svise_synth_full_s60
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-svise-synthetic-full/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 20
#SBATCH --mem=64G

echo "=== SVISE Full Synthetic Dataset (sigma=60, smoothed) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "========================================================="

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Run
cd /home/ka/ka_iai/ka_bj2362/dsr/SVISE/synthetic_dataset_validation
python run_svise_synthetic_full_smoothed.py

echo "End: $(date)"
