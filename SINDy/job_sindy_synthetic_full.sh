#!/bin/bash
#SBATCH --job-name=sindy_synth_full
#SBATCH --time=04:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-sindy-synthetic-full/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4
#SBATCH --mem=64G

echo "=== SINDy Full Synthetic Dataset ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "===================================="

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Run
cd /home/ka/ka_iai/ka_bj2362/dsr/SINDy
python run_sindy_synthetic_full.py

echo "End: $(date)"
