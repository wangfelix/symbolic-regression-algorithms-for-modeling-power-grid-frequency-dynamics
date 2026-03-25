#!/bin/bash
#SBATCH --job-name=svise_synth_all
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-synthetic-noiseless/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH --array=0-17

# ---- Configuration ----
# Active chunks after dead-chunk filtering (~1559 out of 8640 total)
TOTAL_ACTIVE_CHUNKS=1559
N_JOBS=18           # Must match --array range (0 to N_JOBS-1)

# Calculate chunk range for this array task
CHUNKS_PER_JOB=$(( (TOTAL_ACTIVE_CHUNKS + N_JOBS - 1) / N_JOBS ))  # ceiling division
START_CHUNK=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END_CHUNK=$(( START_CHUNK + CHUNKS_PER_JOB - 1 ))

# Clamp end to total chunks
if [ $END_CHUNK -ge $TOTAL_ACTIVE_CHUNKS ]; then
    END_CHUNK=$(( TOTAL_ACTIVE_CHUNKS - 1 ))
fi

echo "Array task $SLURM_ARRAY_TASK_ID: processing active chunks $START_CHUNK to $END_CHUNK"

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the synthetic validation directory
cd /home/ka/ka_iai/ka_bj2362/dsr/svise/synthetic_dataset_validation

# Create log directory if needed
mkdir -p ./slurm-logs-synthetic-noiseless

# Use the base slurm job ID to force all jobs into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

# Run the evaluation script for this chunk range
python -u run_svise_synthetic_noiseless.py --start-chunk $START_CHUNK --end-chunk $END_CHUNK --run-name ${RUN_NAME}
