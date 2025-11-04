#!/bin/bash 
#SBATCH -J latent_gray_intermediate_dataaugmentation
#SBATCH -o ../slurm_oe/%x_%j-%a.out
#SBATCH -e ../slurm_oe/%x_%j-%a.err
#SBATCH -p youlab-gpu 
#SBATCH --array=1-10 
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -euo pipefail
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/seed_to_sim1_sim2_deterministic
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python latent_generation/latent_intermediate_dataaugmentation.py