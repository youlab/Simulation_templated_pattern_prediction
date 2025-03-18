#!/bin/bash 
#SBATCH -o slurm_latent_75k_intermediate.out
#SBATCH -e slurm_latent_75k_intermediate.err
#SBATCH -p youlab-gpu,scavenger-gpu,gpu-common
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy
cd /hpc/group/youlab/ks723/miniconda3/Lingchong/Latent_generation
python Latent_from_final_patterns_intermediate.py