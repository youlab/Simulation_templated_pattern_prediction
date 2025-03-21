#!/bin/bash 
#SBATCH -o slurm_Patterns_3Tp2Cond.out
#SBATCH -e slurm_Patterns_3Tp2Cond.err
#SBATCH -p common-old,scavenger,youlab
#SBATCH --array=1-400
#SBATCH --mem=3G
#SBATCH --mail-type=ALL
module load Matlab/R2022a
cd /hpc/home/ks723/Test/MATLAB_Scripts
matlab -nodisplay -singleCompThread -r Optimal_Patterns_3Tp2Cond
# blank line