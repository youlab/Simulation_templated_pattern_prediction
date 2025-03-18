#!/bin/bash 
#SBATCH -o slurm_latent_SimcorrtoExp.out
#SBATCH -e slurm_latent_SimcorrtoExp.err
#SBATCH -p youlab-gpu,scavenger-gpu,gpu-common
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy
cd /hpc/group/youlab/ks723/miniconda3/Lingchong/Latent_generation
python latent_from_SimcorrtoExp_images.py