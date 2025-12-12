#!/bin/bash 
#SBATCH -o slurm_CMMD_20251011.out
#SBATCH -e slurm_CMMD_20251011.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/cmmd # git clone from https://github.com/sayakpaul/cmmd-pytorch
# Folder pairs
declare -A comparisons
comparisons["exp_vs_sim"]=" /hpc/group/youlab/ks723/storage/Processed_testsets/exp /hpc/group/youlab/ks723/storage/Processed_testsets/sim"
comparisons["exp_vs_pred"]="/hpc/group/youlab/ks723/storage/Processed_testsets/exp /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v2025926_1251_simtoexp_v3/"
comparisons["sim_vs_pred"]="/hpc/group/youlab/ks723/storage/Processed_testsets/sim /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v2025926_1251_simtoexp_v3/"
# comparisons["exp_test_vs_exp_train"]=" /hpc/group/youlab/ks723/storage/Processed_testsets/exp /hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_preprocess_png/"
comparisons["exp_vs_pred_seed"]=" /hpc/group/youlab/ks723/storage/Processed_testsets/exp /hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v20251011_841_seedtoexp_swapped_v3/"


# Loop and run
for label in "${!comparisons[@]}"; do
    read dir1 dir2 <<< "${comparisons[$label]}"
    echo "Running $label: $dir1 vs $dir2"
    cmmd_output=$(python main.py "$dir1" "$dir2" --batch_size=1)
    echo "CMMD ($label) = $cmmd_output"
done

