#!/bin/bash 
#SBATCH -o slurm_ControlNet_simtoexp_20251015_%a.out
#SBATCH -e slurm_ControlNet_simtoexp_20251015_%a.err
#SBATCH -p youlab-gpu
#SBATCH --array=1-5
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python simtoexp_train.py