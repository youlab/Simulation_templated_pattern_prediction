#!/bin/bash 
#SBATCH -o slurm_Patterns_ExptoSim_Fixed.out
#SBATCH -e slurm_Patterns_ExptoSim_Fixed.err
#SBATCH -p common-old,scavenger,youlab
#SBATCH --mem=3G
#SBATCH --mail-type=ALL
module load Matlab/R2022a
cd /hpc/home/ks723/Test/MATLAB_Scripts
matlab -nodisplay -singleCompThread -r Optimal_Patterns_ExperimentalCondns_Fixedgrid
# blank line