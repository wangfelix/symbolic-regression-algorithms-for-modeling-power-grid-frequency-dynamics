#!/bin/bash
#SBATCH --job-name=pysr_hp_5min
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-hp-5min/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hr7224@partner.kit.edu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-59

# 60 Hyperparameter-Kombinationen → array 0-59
# Jeder Job wertet genau eine Kombination aus (--combo-index)

COMBO_INDEX=$SLURM_ARRAY_TASK_ID

echo "=============================================="
echo "SLURM Array Task : $SLURM_ARRAY_TASK_ID"
echo "Combo Index      : $COMBO_INDEX"
echo "Host             : $(hostname)"
echo "Start            : $(date)"
echo "=============================================="

mkdir -p ./slurm-logs-hp-5min

# Activate virtual environment
source /home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation/venv/bin/activate

# Change to project directory
cd /home/ka/ka_iai/ka_hr7224/synthetic_dataset_validation

python -u run_pysr_hp_tuning_5min.py \
    --combo-index $COMBO_INDEX 

echo "=============================================="
echo "End : $(date)"
echo "=============================================="
