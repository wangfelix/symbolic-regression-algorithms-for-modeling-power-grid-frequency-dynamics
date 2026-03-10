#!/bin/bash
#SBATCH --job-name=svise_all
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-5min-all-chunks/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 70
#SBATCH --mem=32G
#SBATCH --array=0-31

# ---- Configuration ----
TOTAL_CHUNKS=31760
N_JOBS=32  # Must match --array range (0 to N_JOBS-1)

# Calculate chunk range for this array task
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))  # ceiling division
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
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

# Run the evaluation script for this chunk range
python -u run_analysis_5min_all_chunks.py --start-chunk $START_CHUNK --end-chunk $END_CHUNK
