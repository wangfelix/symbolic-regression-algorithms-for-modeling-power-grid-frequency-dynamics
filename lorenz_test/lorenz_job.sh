#!/bin/bash
#SBATCH --job-name=pdf_symbolic_regression_lorenz_attractor
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4 # 2 CPUs for parallelism

# Activate virtual environment
source /home/ka/ka_iai/ka_bj2362/dsr/venv/bin/activate

# Change to the project root directory
cd /home/ka/ka_iai/ka_bj2362/dsr/lorenz_test

# Run the python script
python -u  run_lorenz.py