"""
Figure 1: Display experimental patterns, seeds, and simulations
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "seed_to_sim1_sim2_deterministic"))
sys.path.insert(0, str(REPO_DIR))

from models.vae import encode_img, decode_img
from utils.preprocess import (  
    scale_latents, 
    preprocess_simulation_output_data,
    preprocess_experimental_backgroundwhite_rawfiles,
    preprocess_simulation_input_data
)
from config_automate import SEED_FOLDER, EXPERIMENTAL_FOLDER, SIMULATED_FOLDER
from config_automate import FILES_FIG1A, FILES_FIG1B


def generate_fig1a(output_dir):
    """
    Generate Figure 1A: Seed, Simulation, and Experimental pattern comparison
    """
    # Get file lists
    seed_files = []
    for f in FILES_FIG1A:
        base = os.path.splitext(f)[0]      # e.g. "Fixed_19_6"
        _, x, _ = base.split('_')           # x == "19"
        seed_name = f"Input_Fixed_{x}.png"
        seed_files.append(seed_name)

    # Process images
    seed_data = preprocess_simulation_input_data(SEED_FOLDER, 0, len(seed_files), img_filenames=seed_files)
    seed_images = [data[0] for data in seed_data]
    seed_images = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape)==2 else img for img in seed_images]

    sim_data = preprocess_simulation_output_data(SIMULATED_FOLDER, 0, len(FILES_FIG1A), img_filenames=FILES_FIG1A)
    sim_images = [data[0] for data in sim_data]

    exp_data = preprocess_experimental_backgroundwhite_rawfiles(EXPERIMENTAL_FOLDER, 0, len(FILES_FIG1A), img_filenames=FILES_FIG1A)
    exp_images = exp_data

    # Plot
    fig, axes = plt.subplots(3, len(FILES_FIG1A), figsize=(2*len(FILES_FIG1A), 6), layout='constrained')

    for idx in range(len(FILES_FIG1A)):
        axes[0, idx].imshow(seed_images[idx], cmap='gray')
        axes[0, idx].axis('off')
        axes[1, idx].imshow(sim_images[idx], cmap='gray')
        axes[1, idx].axis('off')
        axes[2, idx].imshow(exp_images[idx])
        axes[2, idx].axis('off')

    # Save instead of show
    output_path = Path(output_dir) / "fig1a_seed_sim_exp.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def generate_fig1b_sim(output_dir):
    """
    Generate Figure 1B (Simulation): VAE encoding/reconstruction of simulated patterns
    """
    # Process simulation images
    sim_data = preprocess_simulation_output_data(SIMULATED_FOLDER, 0, len(FILES_FIG1B), img_filenames=FILES_FIG1B)
    sim_images = [data[0] for data in sim_data]
    sim_images = np.array(sim_images)

    # Convert grayscale to RGB and normalize
    sim_rgb = np.stack([sim_images, sim_images, sim_images], axis=-1)
    sim_rgb = np.transpose(sim_rgb, (0, 3, 1, 2)) / 255.0

    # Convert to tensor and encode
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    X = torch.Tensor(sim_rgb)

    # Encode
    encoded_latents = []
    for i in range(X.shape[0]):
        latent = encode_img(X[i:i+1].to(device))
        encoded_latents.append(latent.cpu())

    latents = torch.cat(encoded_latents, dim=0)
    latents_scaled = scale_latents(latents)
    reconstructed = decode_img(latents)

    # Save visualization (modified display_predicted_images to save instead of show)
    output_path = Path(output_dir) / "fig1b_sim_reconstruction.png"
    _save_predicted_images(X, latents_scaled, reconstructed, 5, output_path)
    return output_path


def generate_fig1b_exp(output_dir):
    """
    Generate Figure 1B (Experimental): VAE encoding/reconstruction of experimental patterns
    """
    # Process experimental images
    exp_data = preprocess_experimental_backgroundwhite_rawfiles(EXPERIMENTAL_FOLDER, 0, len(FILES_FIG1B), img_filenames=FILES_FIG1B)
    exp_images = np.array(exp_data)

    # Normalize and prepare for VAE
    exp_images = np.transpose(exp_images, (0, 3, 1, 2)) / 255.0
    X = torch.Tensor(exp_images)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Encode
    encoded_latents = []
    for i in range(X.shape[0]):
        latent = encode_img(X[i:i+1].to(device))
        encoded_latents.append(latent.cpu())

    latents = torch.cat(encoded_latents, dim=0)
    latents_scaled = scale_latents(latents)
    reconstructed = decode_img(latents)

    # Save visualization
    output_path = Path(output_dir) / "fig1b_exp_reconstruction.png"
    _save_predicted_images(X[0:5], latents_scaled[0:5], reconstructed[0:5], 5, output_path)
    return output_path


def _tensor_to_pil(tensor):
    """Convert tensor to PIL image (same as tensor_to_pil_v2 from utils.display)"""
    tensor = tensor.permute(1, 2, 0)  # Convert to (height, width, channels)
    img = (tensor.cpu().numpy() * 255).astype('uint8')
    return img.squeeze()


def _save_predicted_images(original, latents_scaled, reconstructed, num_samples, output_path):
    """
    Helper function to save predicted images (modified from display_predicted_images)
    Saves figure instead of showing it
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*2, 6))
    
    for i in range(num_samples):
        # Original - convert using tensor_to_pil
        image_o = _tensor_to_pil(original[i])
        axes[0, i].imshow(image_o, cmap='gray')
        axes[0, i].axis('off')
        
        # Latent (scaled visualization) - convert using tensor_to_pil
        image_l = _tensor_to_pil(latents_scaled[i])
        axes[1, i].imshow(image_l, cmap='gray')
        axes[1, i].axis('off')
        
        # Reconstructed - convert using tensor_to_pil
        image_r = _tensor_to_pil(reconstructed[i])
        axes[2, i].imshow(image_r, cmap='gray')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_fig1(output_dir, **kwargs):
    """
    Generate all components of Figure 1
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    paths.append(generate_fig1a(output_dir))
    paths.append(generate_fig1b_sim(output_dir))
    paths.append(generate_fig1b_exp(output_dir))
    
    return paths
