#!/bin/bash
#SBATCH --job-name=svise_analysis
#SBATCH --time=5:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate

# Change to the project root directory
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

# Run the python script
#python -u run_analysis.py
python -u run_analysis.py --model integrator --sigma 5 --epochs 30000 --degree 2 --tau 1e-2
