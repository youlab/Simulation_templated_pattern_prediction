#!/bin/bash 
#SBATCH -o slurm_batch_infer_ablation_20251023.out
#SBATCH -e slurm_batch_infer_ablation_20251023.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate test_pytorch_ipy_v2
cd /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential
python batch_infer_ablation.py "$@"