#!/bin/bash 
#SBATCH -o slurm_Optimal_Patterns_3Tp2Cond_ModelTesting.out
#SBATCH -e slurm_Optimal_Patterns_3Tp2Cond_ModelTesting.err
#SBATCH -p common-old,scavenger,youlab
#SBATCH --mem=3G
#SBATCH --array=1-8
#SBATCH --mail-type=ALL
module load Matlab/R2022a
cd /hpc/home/ks723/Test/MATLAB_Scripts
matlab -nodisplay -singleCompThread -r Optimal_Patterns_3Tp2Cond_ModelTesting
# blank line