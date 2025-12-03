"""
Automated configuration file for reproducing figures.

INSTRUCTIONS FOR USERS:
1. Set DATA_DIR to your desired data storage location (the ONLY path you need to change)
2. Run setup_data.py to download and prepare all datasets
3. All other paths will automatically reference the correct locations

After setup_data.py completes, the directory structure will be:

DATA_DIR/
├── extracted/
│   ├── seed_to_sim_deterministic/
│   │   ├── Sim_050924_seed/
│   │   ├── Sim_050924_ModelTesting_seed/
│   │   ├── Sim_050924_intermediate_Tp3/
│   │   ├── Sim_050924_ModelTesting_intermediate_Tp3/
│   │   ├── Sim_050924_complex_Tp3/
│   │   ├── Sim_050924_ModelTesting_complex_Tp3/
│   │   ├── saved_models/              # Trained model checkpoints
│   │   │   ├── Pixel_32x32x3to32x32x4_dilRESNET_30k_graypatterns_seedtointermediate_v101_4-*.pt
│   │   │   ├── Pixel_32x32x3to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_Model_30000_v101_*.pt
│   │   │   ├── models_Fig4/
│   │   │   └── models_dataaugmentation_Fig4/
│   │   ├── logs_selected/             # Training logs
│   │   └── ARIAL.TTF
│   ├── sim_to_exp_diffusion/
│   │   ├── Exp_SimcorrtoExp_seed/
│   │   ├── Exp/
│   │   ├── SimcorrtoExp/
│   │   ├── Exp_SimcorrtoExp_testset_seed/
│   │   ├── SimcorrtoExp_testset/
│   │   └── Exp_testset/
│   └── latents/
│       ├── latents_intermediate.pickle
│       └── latents_complex.pickle
"""

from pathlib import Path
import matplotlib as mpl

# ==============================================================================
# BASE CONFIGURATION - ONLY PATH USERS NEED TO CHANGE
# ==============================================================================

DATA_DIR = Path("/hpc/group/youlab/ks723/Physics_constrained_DL_pattern_prediction/data")  # <-- Change this to your data storage location

# Repository root (auto-detected)
REPO_DIR = Path(__file__).resolve().parent

# ==============================================================================
# DERIVED PATHS - Automatically built from DATA_DIR
# ==============================================================================

# Extracted datasets
EXTRACTED_DIR = DATA_DIR / "extracted"
SEED_TO_SIM_DIR = EXTRACTED_DIR / "seed_to_sim_deterministic"
SIM_TO_EXP_DIR = EXTRACTED_DIR / "sim_to_exp_diffusion"
LATENTS_DIR = EXTRACTED_DIR / "latents"

# Models directory (trained models are in extracted data)
SEED_TO_SIM_MODELS_DIR = SEED_TO_SIM_DIR / "saved_models"

# Repository paths for SD VAE and ControlNet checkpoints
# CONTROLNET_DIR = REPO_DIR / "sim_to_exp_diffusion" / "controlnet_essential"

# ==============================================================================
# FIGURE 1: Seed/Simulation/Experiment Comparison & VAE Reconstruction
# ==============================================================================

# Input folders
SEED_FOLDER = str(SIM_TO_EXP_DIR / "Exp_SimcorrtoExp_seed")                    # Exp_SimcorrtoExp_seed.tar
EXPERIMENTAL_FOLDER = str(SIM_TO_EXP_DIR / "Exp")   # Exp.tar
SIMULATED_FOLDER = str(SIM_TO_EXP_DIR / "SimcorrtoExp")                     # SimcorrtoExp.tar

# Selected files for Fig 1a and 1b
FILES_FIG1A = ['Fixed_19_6.TIF', 'Fixed_15_3.TIF', 'Fixed_25_4.TIF', 'Fixed_6_1.TIF', 'Fixed_26_3.TIF']
FILES_FIG1B = ['5_2.TIF', 'Fixed_14_2.TIF', 'Fixed_29_3.TIF', 'Fixed_15_2.TIF', 'Fixed_22_1.TIF']

# ==============================================================================
# FIGURE 2: Seed to Intermediate Pattern Prediction (dilResNet)
# ==============================================================================

# Training/test data folders
SIMULATION_TEST_INPUT_FOLDER_2 = str(SEED_TO_SIM_DIR / "Sim_050924_seed")          # Sim_050924_seed.tar
SIMULATION_TEST_OUTPUT_FOLDER_2= str(SEED_TO_SIM_DIR / "Sim_050924_intermediate_Tp3") # Sim_050924_intermediate_Tp3.tar
SIMULATION_TEST_INPUT_FOLDER = str(SEED_TO_SIM_DIR / "Sim_050924_ModelTesting_seed")          # Sim_050924_ModelTesting_seed.tar
SIMULATION_TEST_OUTPUT_FOLDER = str(SEED_TO_SIM_DIR / "Sim_050924_ModelTesting_intermediate_Tp3") # Sim_050924_ModelTesting_intermediate_Tp3.tar

# Latents and model
LATENT_OUTPUT_SAVED = str(LATENTS_DIR / "latents_intermediate.pickle")
MODEL_DILRESNET_FIG2 = str(SEED_TO_SIM_MODELS_DIR / "Pixel_32x32x3to32x32x4_dilRESNET_30k_graypatterns_seedtointermediate_v101_4-1759366230_best.pt")

# ==============================================================================
# FIGURE 3: Intermediate to Complex Pattern Prediction (dilResNet)
# ==============================================================================

# Training/test data folders (note: uses intermediate folders as input, complex as output)
SIMULATION_TEST_OUTPUT_FOLDER_TDB = str(SEED_TO_SIM_DIR / "Sim_050924_ModelTesting_complex_Tp3")   # Sim_050924_ModelTesting_complex_Tp3.tar
SIMULATION_TEST_OUTPUT_FOLDER_TDB_2 = str(SEED_TO_SIM_DIR / "Sim_050924_complex_Tp3") # Sim_050924_complex_Tp3.tar

# Latents and model
LATENT_OUTPUT_TDB_SAVED = str(LATENTS_DIR / "latents_complex.pickle")
MODEL_DILRESNET_FIG3 = str(SEED_TO_SIM_MODELS_DIR / "Pixel_32x32x3to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_Model_30000_v101_Cluster_GPU_tfData-1759363890_best.pt")

# ==============================================================================
# FIGURE 4: Data Efficiency Analysis (Training Set Size vs Performance)
# ==============================================================================

# Uses same data folders as Fig 3
# Requires multiple model checkpoints and loss logs
MODEL_SAVE_LOCATION = str(SEED_TO_SIM_MODELS_DIR)  # Directory containing all model checkpoints
LOGS_SAVE_LOCATION = str(SEED_TO_SIM_DIR / "logs_selected")  # Directory containing training logs

# Font path for plotting (optional - set to None to use default matplotlib fonts)
# To use custom font, set this to the full path to your .ttf font file

#########
# Font for plots
FPATH = Path(mpl.get_data_path(), str(SEED_TO_SIM_DIR / "ARIAL.TTF"))  # ARIAL.TTF
#########
# ==============================================================================
# FIGURE 5: Simulation to Experiment Translation (ControlNet)
# ==============================================================================

# Test data folders
SIM_FOLDER_TEST = str(SIM_TO_EXP_DIR / "SimcorrtoExp_testset")        # SimcorrtoExp_testset.tar
SEED_FOLDER_TEST = str(SIM_TO_EXP_DIR / "Exp_SimcorrtoExp_testset_seed") # Exp_SimcorrtoExp_testset_seed.tar
EXP_FOLDER_TEST = str(SIM_TO_EXP_DIR / "Exp_testset")    # Exp_testset.tar

# ControlNet checkpoint (from HuggingFace dataset)
CKPT_PATH = str(SIM_TO_EXP_DIR / "checkpoint_simtoexp" / "epoch=4-step=51124.ckpt")

# Temporary output directory for inference (used internally by fig5)
OUTPUT_DIR_SIMTOEXP = str(Path(__file__).parent / "inference" / "temp_simtoexp")

# ==============================================================================
# VALIDATION: Check if setup has been completed
# ==============================================================================

def validate_setup():
    """
    Check if setup_data.py has been run and all required paths exist.
    Returns (bool, list of missing paths).
    """
    missing = []
    
    # Check critical directories
    critical_dirs = [
        EXTRACTED_DIR,
        SEED_TO_SIM_DIR,
        SIM_TO_EXP_DIR,
        LATENTS_DIR,
        SEED_TO_SIM_MODELS_DIR
    ]
    
    for path in critical_dirs:
        if not Path(path).exists():
            missing.append(str(path))
    
    # Check critical files
    critical_files = [
        LATENT_OUTPUT_SAVED,
        LATENT_OUTPUT_TDB_SAVED,
        MODEL_DILRESNET_FIG2,
        MODEL_DILRESNET_FIG3,
        CKPT_PATH
    ]
    
    for path in critical_files:
        if not Path(path).exists():
            missing.append(str(path))
    
    if missing:
        return False, missing
    return True, []


if __name__ == "__main__":
    # Quick validation when config is run directly
    is_valid, missing_paths = validate_setup()
    if not is_valid:
        print("WARNING: Some required paths are missing.")
        print("Please run setup_data.py first to download and prepare all datasets.")
        print("\nMissing paths:")
        for path in missing_paths[:5]:  # Show first 5
            print(f"  - {path}")
        if len(missing_paths) > 5:
            print(f"  ... and {len(missing_paths) - 5} more")
    else:
        print("Configuration validated successfully.")
        print(f"Data directory: {DATA_DIR}")