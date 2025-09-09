#!/bin/bash 
#SBATCH -o slurm_batch_infer_ablation_20250514.out
#SBATCH -e slurm_batch_infer__ablation_20250514.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy_v2
cd /hpc/dctrl/ks723/Huggingface_repos/ControlNet_repo/controlnet_repo
python batch_infer_ablation.py "$@"