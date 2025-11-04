#!/bin/bash 
#SBATCH -J Training_intermediatetocomplex_dilRESNETs_parallel
#SBATCH -o ../slurm_oe/%x_%j_%a.out
#SBATCH -e ../slurm_oe/%x_%j_%a.err
#SBATCH -p youlab-gpu
#SBATCH --array=1-10
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -euo pipefail
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/seed_to_sim1_sim2_deterministic
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -u Training_intermediatetocomplex_dilResNets.py
