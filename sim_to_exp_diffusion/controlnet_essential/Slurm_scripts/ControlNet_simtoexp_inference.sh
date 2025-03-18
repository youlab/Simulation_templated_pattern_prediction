#!/bin/bash 
#SBATCH -o slurm_ControlNet_simtoexp_inference_20250211.out
#SBATCH -e slurm_ControlNet_simtoexp_inference_20250211.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy_v2
cd /hpc/dctrl/ks723/Huggingface_repos/ControlNet_repo/controlnet_repo
python inference_simtoexp.py