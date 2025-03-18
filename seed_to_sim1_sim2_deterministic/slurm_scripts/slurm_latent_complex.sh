#!/bin/bash 
#SBATCH -o slurm_latent_75k_complex.out
#SBATCH -e slurm_latent_75k_complex.err
#SBATCH -p youlab-gpu,scavenger-gpu,gpu-common
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy
cd /hpc/group/youlab/ks723/miniconda3/Lingchong/Latent_generation
python latent_from_complex.py