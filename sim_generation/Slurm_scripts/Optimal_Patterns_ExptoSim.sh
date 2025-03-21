#!/bin/bash 
#SBATCH -o slurm_Patterns_ExptoSim.out
#SBATCH -e slurm_Patterns_ExptoSim.err
#SBATCH -p common-old,scavenger,youlab
#SBATCH --array=1-100
#SBATCH --mem=3G
#SBATCH --mail-type=ALL
module load Matlab/R2022a
cd /hpc/home/ks723/Test/MATLAB_Scripts
matlab -nodisplay -singleCompThread -r Optimal_Patterns_ExperimentalCondns
# blank line