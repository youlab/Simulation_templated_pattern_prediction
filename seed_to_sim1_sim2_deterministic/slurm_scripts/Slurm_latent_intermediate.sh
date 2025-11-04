#!/bin/bash 
#SBATCH -J latent_gray_intermediate
#SBATCH -o ../slurm_oe/%x_%j.out
#SBATCH -e ../slurm_oe/%x_%j.err
#SBATCH -p youlab-gpu,scavenger-gpu,gpu-common
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -euo pipefail
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/seed_to_sim1_sim2_deterministic
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python latent_generation/Latent_from_final_patterns_intermediate.py


