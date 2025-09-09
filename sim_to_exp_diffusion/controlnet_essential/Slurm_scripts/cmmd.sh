#!/bin/bash 
#SBATCH -o slurm_CMMD_20250604.out
#SBATCH -e slurm_CMMD_20250604.err
#SBATCH -p youlab-gpu
#SBATCH --exclusive
#SBATCH --mem=24G
#SBATCH --mail-type=ALL
source activate pytorch_PA_patternprediction
cd /hpc/dctrl/ks723/cmmd
# Folder pairs
declare -A comparisons
comparisons["exp_vs_sim"]=" /hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3_png/ /hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_v3_png/"
comparisons["exp_vs_pred"]="/hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3_png/ /hpc/dctrl/ks723/inference/Generated_202552_327_simtoexp_v3/"
comparisons["sim_vs_pred"]="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_v3_png/ /hpc/dctrl/ks723/inference/Generated_202552_327_simtoexp_v3/"
comparisons["exp_test_vs_exp_train"]=" /hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3_png/ /hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_preprocess_png/"
comparisons["exp_vs_pred_seed"]=" /hpc/group/youlab/ks723/storage/Exp_images/Final_Test_set_preprocess_v3_png/ /hpc/dctrl/ks723/inference/v2025821_156_seedtoexp_v3/"


# Loop and run
for label in "${!comparisons[@]}"; do
    read dir1 dir2 <<< "${comparisons[$label]}"
    echo "Running $label: $dir1 vs $dir2"
    cmmd_output=$(python main.py "$dir1" "$dir2" --batch_size=1)
    echo "CMMD ($label) = $cmmd_output"
done

