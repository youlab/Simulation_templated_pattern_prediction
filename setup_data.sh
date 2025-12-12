#!/bin/bash 
#SBATCH -o slurm_setup_data_%j.out
#SBATCH -e slurm_setup_data_%j.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL

source ~/.bashrc  
conda activate pytorch_PA_patternprediction
python setup_data.py 