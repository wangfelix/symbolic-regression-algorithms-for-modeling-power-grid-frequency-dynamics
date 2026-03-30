#!/bin/bash
#SBATCH --job-name=pysr_syn_5min
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-synthetic-5minNewParam/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hr7224@partner.kit.edu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --array=0-86

# 8640 Chunks / 100 Jobs = 87 Jobs à ~100 Chunks
TOTAL_CHUNKS=8640
N_JOBS=87
CHUNKS_PER_JOB=$(( (TOTAL_CHUNKS + N_JOBS - 1) / N_JOBS ))

START_CHUNK=$(( SLURM_ARRAY_TASK_ID * CHUNKS_PER_JOB ))
END_CHUNK=$(( START_CHUNK + CHUNKS_PER_JOB ))
if [ $END_CHUNK -gt $TOTAL_CHUNKS ]; then
    END_CHUNK=$TOTAL_CHUNKS
fi

echo "=============================================="
echo "SLURM Array Task : $SLURM_ARRAY_TASK_ID"
echo "Chunks           : $START_CHUNK bis $END_CHUNK"
echo "Host             : $(hostname)"
echo "Start            : $(date)"
echo "=============================================="

mkdir -p ./slurm-logs-synthetic-5min

source /home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/venv/bin/activate
cd /home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation

python -u run_pysr_synthetic_5min.py \
    --start-chunk $START_CHUNK \
    --end-chunk $END_CHUNK

echo "=============================================="
echo "End : $(date)"
echo "=============================================="
