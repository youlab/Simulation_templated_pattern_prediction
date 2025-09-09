#!/bin/bash 
#SBATCH -o slurm_ControlNet_seedtoexp_20250819.out
#SBATCH -e slurm_ControlNet_seedtoexp_20250819.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/Huggingface_repos/ControlNet_repo/controlnet_repo
python seedtoexp_train.py