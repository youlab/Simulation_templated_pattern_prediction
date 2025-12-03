#!/bin/bash 
#SBATCH -J Seed_DataAugmentation
#SBATCH -o ../slurm_oe/%x_%j-%a.out
#SBATCH -e ../slurm_oe/%x_%j-%a.err
#SBATCH -p youlab-gpu 
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --mail-type=ALL

set -euo pipefail
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python Seed_DataAugmentation.py