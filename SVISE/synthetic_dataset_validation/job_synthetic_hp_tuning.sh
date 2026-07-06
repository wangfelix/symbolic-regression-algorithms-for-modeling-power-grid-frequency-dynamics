#!/bin/bash
#SBATCH --job-name=svise_synth_hp
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-synthetic-hp-tuning/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 20
#SBATCH --mem=16G

#SBATCH --array=0-59

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the synthetic validation directory
cd /home/ka/ka_iai/ka_bj2362/dsr/SVISE/synthetic_dataset_validation

# Create log directory if needed
mkdir -p ./slurm-logs-synthetic-hp-tuning

# Use the base slurm job ID to force all jobs into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

# Run the hyperparameter tuning script with array
python -u run_svise_synthetic_hp_tuning.py --n-chunks 100 --n-samples 60 --seed 42 --run-name ${RUN_NAME} --combo-index $SLURM_ARRAY_TASK_ID
