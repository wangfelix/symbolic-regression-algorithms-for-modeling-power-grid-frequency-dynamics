#!/bin/bash
#SBATCH --job-name=sindy_synth
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-synthetic-all-chunks/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --array=0-213

# ---- Configuration ----
# 32,172,226 samples / 300 = 107,240 chunks
TOTAL_CHUNKS=107240
N_JOBS=214  # Must match --array range (0 to N_JOBS-1)

# Calculate chunk range for this array task
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))
START_CHUNK=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END_CHUNK=$(( START_CHUNK + CHUNKS_PER_JOB - 1 ))

# Clamp end to total chunks
if [ $END_CHUNK -ge $TOTAL_CHUNKS ]; then
    END_CHUNK=$(( TOTAL_CHUNKS - 1 ))
fi

echo "Array task $SLURM_ARRAY_TASK_ID: processing chunks $START_CHUNK to $END_CHUNK"

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the project directory
cd /home/ka/ka_iai/ka_bj2362/dsr/SINDy

# Create log directory if needed
mkdir -p ./slurm-logs-synthetic-all-chunks

# Use the base slurm job ID to force all jobs into the same directory
if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    RUN_NAME="run_local_$(date +%Y%m%d_%H%M%S)"
else
    RUN_NAME="run_SLURM_${SLURM_ARRAY_JOB_ID}"
fi

# Run SINDy analysis for this chunk range
python -u run_sindy_synthetic_all_chunks.py \
    --start-chunk $START_CHUNK \
    --end-chunk $END_CHUNK \
    --run-name ${RUN_NAME}
