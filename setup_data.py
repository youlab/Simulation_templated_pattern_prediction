#!/usr/bin/env python3
"""
Setup Data: Download datasets and generate latents

This script performs the one-time setup process:
1. Download datasets from HuggingFace
2. Extract tar files
3. Generate VAE latents for intermediate and complex patterns
4. Download SD 1.5 checkpoint and attach ControlNet

Usage:
    python setup_data.py

Prerequisites:
    - Edit DATA_DIR in config_automate.py to your desired data storage location
    - Ensure you have sufficient disk space (~20GB)
    - GPU recommended for latent generation
"""

import tarfile
import os
import shutil
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
from huggingface_hub import snapshot_download

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REPO_DIR = Path(__file__).resolve().parent

# Import DATA_DIR from config_automate
sys.path.insert(0, str(REPO_DIR))
from config_automate import DATA_DIR

DATA_DIR = str(DATA_DIR)  # Convert to string for compatibility

# ==============================================================================
# Step 1: Download Datasets from HuggingFace
# ==============================================================================

def download_datasets():
    """Download all required datasets from HuggingFace"""
    print("=" * 70)
    print("STEP 1: Downloading datasets from HuggingFace")
    print("=" * 70)
    
    print(f"Downloading to: {DATA_DIR}")
    print("This may take a while (~19GB)...")
    
    snapshot_download(
        repo_id="HotshotGoku/Simulation_templated_pattern_prediction",
        repo_type="dataset",
        local_dir=DATA_DIR
    )
    
    print("Download complete!\n")


# ==============================================================================
# Step 2: Extract Tar Files
# ==============================================================================

def extract_tar_files():
    """Extract all tar files to organize the data"""
    print("=" * 70)
    print("STEP 2: Extracting tar files")
    print("=" * 70)
    
    extract_dir = os.path.join(DATA_DIR, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract tar files from seed_to_sim_deterministic folder
    print("\nExtracting seed_to_sim_deterministic files...")
    seed_sim_dir = os.path.join(DATA_DIR, "seed_to_sim_deterministic")
    seed_sim_extract = os.path.join(extract_dir, "seed_to_sim_deterministic")
    os.makedirs(seed_sim_extract, exist_ok=True)
    
    for file in os.listdir(seed_sim_dir):
        src = os.path.join(seed_sim_dir, file)
        if file.endswith(".tar"):
            output_dir = os.path.join(seed_sim_extract, file.replace(".tar", ""))
            print(f"  Extracting {file}...")
            with tarfile.open(src) as tar:
                tar.extractall(output_dir)
        else:
            print(f"  Copying {file}...")
            shutil.copy2(src, os.path.join(seed_sim_extract, file))
    
    # Extract tar files from sim_to_exp_diffusion folder
    print("\nExtracting sim_to_exp_diffusion files...")
    sim_exp_dir = os.path.join(DATA_DIR, "sim_to_exp_diffusion")
    sim_exp_extract = os.path.join(extract_dir, "sim_to_exp_diffusion")
    os.makedirs(sim_exp_extract, exist_ok=True)
    
    for file in os.listdir(sim_exp_dir):
        src = os.path.join(sim_exp_dir, file)
        if file.endswith(".tar"):
            output_dir = os.path.join(sim_exp_extract, file.replace(".tar", ""))
            print(f"  Extracting {file}...")
            with tarfile.open(src) as tar:
                tar.extractall(output_dir)
        else:
            print(f"  Copying {file}...")
            shutil.copy2(src, os.path.join(sim_exp_extract, file))
    
    print("Extraction complete!\n")


# ==============================================================================
# Step 3: Generate VAE Latents
# ==============================================================================

def generate_latents(input_dir, output_file, chunk_size=1000, batch_size=96):
    """Generate VAE latents for simulation images"""
    
    # Add repository to path for imports
    sys.path.insert(0, str(REPO_DIR / "seed_to_sim1_sim2_deterministic"))
    
    from models.vae import encode_img
    from utils.preprocess import preprocess_simulation_output_data
    
    print(f"Loading data from: {input_dir}")
    
    all_files = sorted(os.listdir(input_dir))
    total_files = len(all_files)
    print(f"Total files: {total_files}")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    encoded_latents = []
    processed_count = 0
    
    for chunk_start in range(0, total_files, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_files)
        
        output_data = preprocess_simulation_output_data(input_dir, chunk_start, chunk_end)
        if not output_data:
            continue
        
        x_chunk = np.array(output_data).reshape(-1, 1, 256, 256) / 255.0
        x_chunk = torch.Tensor(x_chunk)
        
        for i in range(0, x_chunk.shape[0], batch_size):
            end_idx = min(i + batch_size, x_chunk.shape[0])
            batch = x_chunk[i:end_idx]
            
            processed_count += batch.shape[0]
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count}/{total_files}")
            
            latent_batch = encode_img(batch.to(device))
            encoded_latents.append(latent_batch.cpu())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del output_data, x_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    image_np = torch.cat(encoded_latents, dim=0).numpy()
    print(f"  Final shape: {image_np.shape}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(image_np, f)
    
    print(f"  Saved to: {output_file}")


def generate_all_latents():
    """Generate latents for both intermediate and complex patterns"""
    print("=" * 70)
    print("STEP 3: Generating VAE latents")
    print("=" * 70)
    print("This will take a while (GPU recommended)...\n")
    
    # Generate latents for intermediate patterns
    print("Generating latents for intermediate patterns...")
    intermediate_dir = os.path.join(DATA_DIR, "extracted", "seed_to_sim_deterministic", "Sim_050924_intermediate_Tp3")
    intermediate_output = os.path.join(DATA_DIR, "extracted", "latents", "latents_intermediate.pickle")
    generate_latents(intermediate_dir, intermediate_output)
    print()
    
    # Generate latents for complex patterns
    print("Generating latents for complex patterns...")
    complex_dir = os.path.join(DATA_DIR, "extracted", "seed_to_sim_deterministic", "Sim_050924_complex_Tp3")
    complex_output = os.path.join(DATA_DIR, "extracted", "latents", "latents_complex.pickle")
    generate_latents(complex_dir, complex_output)
    
    print("Latent generation complete!\n")


# ==============================================================================
# Step 4: Setup ControlNet
# ==============================================================================

def setup_controlnet():
    """Download SD checkpoint and attach ControlNet"""
    print("=" * 70)
    print("STEP 4: Setting up ControlNet")
    print("=" * 70)
    
    from huggingface_hub import hf_hub_download
    
    # Define paths
    controlnet_dir = REPO_DIR / "sim_to_exp_diffusion" / "controlnet_essential"
    sd_checkpoint_path = controlnet_dir / "models" / "v1-5-pruned.ckpt"
    controlnet_ini_path = controlnet_dir / "models" / "control_sd15_ini.pth"
    tool_script = controlnet_dir / "tool_add_control.py"
    
    # Download Stable Diffusion checkpoint if not exists
    if not sd_checkpoint_path.exists():
        print(f"\nDownloading Stable Diffusion checkpoint...")
        print(f"  Destination: {sd_checkpoint_path}")
        
        sd_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        hf_hub_download(
            repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            filename="v1-5-pruned.ckpt",
            local_dir=str(sd_checkpoint_path.parent),
            local_dir_use_symlinks=False
        )
        print("  Download complete!")
    else:
        print(f"\nStable Diffusion checkpoint already exists: {sd_checkpoint_path}")
    
    # Check if control_sd15_ini.pth already exists
    if controlnet_ini_path.exists():
        print(f"ControlNet checkpoint already attached: {controlnet_ini_path}")
        print("Setup complete!\n")
        return
    
    # Run tool_add_control.py to attach ControlNet
    print(f"\nAttaching ControlNet to Stable Diffusion...")
    print(f"  Running: {tool_script}")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(tool_script), str(sd_checkpoint_path), str(controlnet_ini_path)],
        cwd=str(controlnet_dir),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: ControlNet attachment failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError("Failed to attach ControlNet")
    
    print(f"  ControlNet attached successfully!")
    print(f"  Output: {controlnet_ini_path}")
    print("Setup complete!\n")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run all setup steps"""
    print("\n" + "=" * 70)
    print("PHYSICS-CONSTRAINED DL PATTERN PREDICTION - DATA SETUP")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Repository directory: {REPO_DIR}")
    print("\nThis script will:")
    print("  1. Download datasets from HuggingFace (~19GB)")
    print("  2. Extract tar files (~22 min)")
    print("  3. Generate VAE latents (~43 min per dataset, GPU recommended)")
    print("  4. Download SD checkpoint and attach ControlNet (~5 min)")
    print("\nTotal estimated time: ~2-3 hours")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return
    
    try:
        # Step 1: Download datasets
        download_datasets()
        
        # Step 2: Extract tar files
        extract_tar_files()
        
        # Step 3: Generate latents
        generate_all_latents()
        
        # Step 4: Setup ControlNet
        setup_controlnet()
        
        print("=" * 70)
        print("SETUP COMPLETE!")
        print("=" * 70)
        print("\nYou can now run: python reproduce_figures.py")
        print("Or train models using the prepared data in: {}".format(DATA_DIR))
        print()
        
    except Exception as e:
        print(f"\nERROR during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()