#!/bin/bash 
#SBATCH -o slurm_ControlNet_simtoexp_BnW_20250320.out
#SBATCH -e slurm_ControlNet_simtoexp_BnW_20250320.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
python simtoexp_BnW_train.py