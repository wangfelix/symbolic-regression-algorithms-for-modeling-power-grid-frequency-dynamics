#!/bin/bash
#SBATCH --job-name=sindy_full_hp
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-sindy-synthetic-full-hp/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4
#SBATCH --mem=16G

# 3 sigma * 3 degree * 5 threshold = 45 combinations
#SBATCH --array=0-44

echo "=== SINDy Full HP Tuning ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "============================"

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Create log directory if needed
mkdir -p /home/ka/ka_iai/ka_bj2362/dsr/SINDy/slurm-logs-sindy-synthetic-full-hp

cd /home/ka/ka_iai/ka_bj2362/dsr/SINDy

# Use the base slurm job ID to force all array tasks into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

python -u run_sindy_synthetic_full_hp_tuning.py \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --run-name ${RUN_NAME}

echo "End: $(date)"
