#!/bin/bash
#SBATCH --job-name=sindy_hp
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-sindy-hp-tuning/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --array=0-59

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the project directory
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

# Use the base slurm job ID to force all jobs into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

# Run SINDy HP tuning for this combo index
# 60 combos total (4 sigma * 3 degree * 5 threshold), one per array task
python -u run_sindy_hp_tuning.py \
    --max-chunks 300 \
    --seed 42 \
    --run-name ${RUN_NAME} \
    --combo-index $SLURM_ARRAY_TASK_ID
