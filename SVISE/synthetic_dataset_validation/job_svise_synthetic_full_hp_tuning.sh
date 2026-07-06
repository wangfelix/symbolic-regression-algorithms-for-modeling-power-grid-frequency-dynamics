#!/bin/bash
#SBATCH --job-name=svise_full_hp
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-svise-synthetic-full-hp/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 20
#SBATCH --mem=64G

# 60 random combos
#SBATCH --array=0-59

echo "=== SVISE Full HP Tuning (sigma=0) ==="
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "======================================="

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Create log directory if needed
mkdir -p /home/ka/ka_iai/ka_bj2362/dsr/SVISE/synthetic_dataset_validation/slurm-logs-svise-synthetic-full-hp

cd /home/ka/ka_iai/ka_bj2362/dsr/SVISE/synthetic_dataset_validation

# Use the base slurm job ID to force all array tasks into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

python -u run_svise_synthetic_full_hp_tuning.py \
    --combo-index $SLURM_ARRAY_TASK_ID \
    --run-name ${RUN_NAME}

echo "End: $(date)"
