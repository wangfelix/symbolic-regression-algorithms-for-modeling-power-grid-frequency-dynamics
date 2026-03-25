#!/bin/bash
#SBATCH --job-name=1dlkm_5m
#SBATCH --time=06:00:00
#SBATCH --partition=cpu
#SBATCH --output=./slurm-logs-1dlkm-reference/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bj2362@partner.kit.edu
#SBATCH -c 4
#SBATCH --mem=16G

source /home/ka/ka_iai/ka_bj2362/dsr/.venv_svise/bin/activate
cd /home/ka/ka_iai/ka_bj2362/dsr/svise

mkdir -p ./slurm-logs-1dlkm-reference

python -u compute_1dlkm_reference_rmse.py --chunk-minutes 5 --sigma 60 --run-name run_SLURM_${SLURM_JOB_ID}
