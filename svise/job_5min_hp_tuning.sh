#!/bin/bash
#SBATCH --job-name=svise_hp_tune
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-5min-hp-tuning/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 16
#SBATCH --mem=64G

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the project root directory
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

# Run the hyperparameter tuning script
# --max-chunks: number of 9:00-10:00 chunks to evaluate per combo (default: 5000)
# --dry-run: quick sanity check with 2 chunks and 5 epochs
#python -u run_analysis_5min_hyperparameter_tuning.py --max-chunks 3
python -u run_analysis_5min_hyperparameter_tuning.py --max-chunks 5 --n-samples 60 --seed 42

