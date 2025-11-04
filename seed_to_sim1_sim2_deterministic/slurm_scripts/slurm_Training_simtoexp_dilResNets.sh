#!/bin/bash 
#SBATCH -J Training_simtoexp_dilRESNETs
#SBATCH -o ../slurm_oe/%x_%j.out
#SBATCH -e ../slurm_oe/%x_%j.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -euo pipefail
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/seed_to_sim1_sim2_deterministic
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -u Training_simtoexp_dilResNets.py
