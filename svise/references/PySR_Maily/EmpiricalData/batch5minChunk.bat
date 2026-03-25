#!/bin/bash
#SBATCH --job-name=pysr_per_chunk
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-5minChunk/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hr7224@partner.kit.edu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-499

# ---- Configuration ----
TOTAL_CHUNKS=107059  # total valid chunks in dataset
N_JOBS=500           # must match --array range (0 to N_JOBS-1)

# Calculate chunk range for this array task
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))   # ceiling division
START_CHUNK=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END_CHUNK=$(( START_CHUNK + CHUNKS_PER_JOB ))                 # exclusive end (Python-style)

# Clamp end to total chunks
if [ $END_CHUNK -gt $TOTAL_CHUNKS ]; then
    END_CHUNK=$TOTAL_CHUNKS
fi

echo "Array task $SLURM_ARRAY_TASK_ID: processing chunks $START_CHUNK to $END_CHUNK"

# Create log directory if it doesn't exist
mkdir -p ./slurm-5minChunk

# Activate virtual environment
source /home/ka/ka_iai/ka_hr7224/PySRCurrent/venv/bin/activate

# Change to project directory
cd /home/ka/ka_iai/ka_hr7224/PySRCurrent

# Run training script for this chunk range
python -u pysr_full_training.py \
    --start-chunk $START_CHUNK \
    --end-chunk $END_CHUNK \
    > slurm-5minChunk/pysr_chunks_${START_CHUNK}_${END_CHUNK}_$(date +%Y%m%d_%H%M%S).log 2>&1