"""
Figure 2: Seed to intermediate pattern prediction with dilResNet
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "seed_to_sim1_sim2_deterministic"))
sys.path.insert(0, str(REPO_DIR))

from models.vae import decode_img
from models.dilResNet import PDEArenaDilatedResNet
import pickle

from config_automate import SIMULATION_TEST_INPUT_FOLDER, SIMULATION_TEST_OUTPUT_FOLDER
from config_automate import SIMULATION_TEST_INPUT_FOLDER_2, SIMULATION_TEST_OUTPUT_FOLDER_2
from config_automate import LATENT_OUTPUT_SAVED, MODEL_DILRESNET_FIG2

from utils.preprocess import preprocess_simulation_input_data, preprocess_simulation_output_data, scale_latents


def _tensor_to_pil(tensor):
    """Convert tensor to PIL image (same as tensor_to_pil_v2 from utils.display)"""
    tensor = tensor.permute(1, 2, 0)  # Convert to (height, width, channels)
    img = (tensor.cpu().numpy() * 255).astype('uint8')
    return img.squeeze()


def _save_predicted_images(input_seed, final_patterns, pred_images, num_samples, output_path, order=[0,1,2]):
    """Save predicted images to file instead of displaying"""
    fig, ax = plt.subplots(3, num_samples, figsize=(6*num_samples/3, 6), layout='constrained')

    for i in range(num_samples):
        image_i = _tensor_to_pil(input_seed[i,:,:,:])
        image_o = _tensor_to_pil(final_patterns[i,:,:,:])
        image_p = _tensor_to_pil(pred_images[i,:,:,:].to("cpu"))
      
        image_list = [image_i, image_o, image_p]
        
        ax[0,i].imshow(image_list[order[0]], cmap='gray')
        ax[0,i].axis('off')

        ax[1,i].imshow(image_list[order[1]], cmap='gray')
        ax[1,i].axis('off')

        ax[2,i].imshow(image_list[order[2]], cmap="gray")
        ax[2,i].axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_fig2(output_dir, **kwargs):
    """
    Generate Figure 2: Seed to intermediate pattern prediction
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # prediction from testing dataset
    
    '''
    Loading test data for model testing
    '''
    
    path_output = SIMULATION_TEST_OUTPUT_FOLDER
    path_output_2 = SIMULATION_TEST_OUTPUT_FOLDER_2
    path_input = SIMULATION_TEST_INPUT_FOLDER
    path_input_2 = SIMULATION_TEST_INPUT_FOLDER_2
    
    # output data
    start_index = 0  
    end_index = 8   
    output_data = preprocess_simulation_output_data(path_output, start_index, end_index)


    ## add some of the images from the main dataset that hasn't been used to train the network 
    
    start_index = 70000  
    end_index = 70025 
    output_data_2 = preprocess_simulation_output_data(path_output_2, start_index, end_index)
    
    output_data.extend(output_data_2)
    # input data
    
    start_index = 0  
    end_index = 8   
    input_data = preprocess_simulation_input_data(path_input, start_index, end_index)


    start_index = 70000  
    end_index = 70025 
    input_data_2 = preprocess_simulation_input_data(path_input_2, start_index, end_index)
    
    input_data.extend(input_data_2)
    
    X = output_data
    y = input_data
    
    
    # size of input data is 32x32 and output data is 256x256
    X = (np.array(X).reshape(-1,1,256,256))
    y = (np.array(y).reshape(-1,1,32,32))

    # normalizing images here to be bw 0 and 1 
    
    X = X/255.0 
    y = y/255.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Convert numpy arrays to torch tensors
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    
    y3 = y.repeat(1, 3, 1, 1)
    y4 = y.repeat(1, 4, 1, 1)

    '''
    Defining dilResNet model for loading model weights
    '''
    
    model = PDEArenaDilatedResNet(
        in_channels=3,               # Input channels
        out_channels=4,              # Output channels 
        hidden_channels=64,          # Number of hidden channels
        num_blocks=15,               # Number of dilated blocks 
        dilation_rates=[1, 2, 4, 8], # Dilation rates for multi-scale feature capture
        activation=nn.ReLU,          # Activation function
        norm=True                    # Use BatchNorm after each convolution
    )

    '''
    Define pre-trained SD VAE for decoding orginal predicted patterns from the latent predicted patterns
    dilResnet will predict latent representation of patterns 
    
    '''
    
    # load trained model weights
    model.load_state_dict(torch.load(MODEL_DILRESNET_FIG2, map_location=device))
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        predicted_latents = model(y3[:,:,:,:].to(device))  #y3 is the input seed with 3 channels for the UNET 
    
    
    predicted_latents_rescaled = predicted_latents
    
    # use the vae decoder to convert the encoded images to final patterns
    
    pred_images = decode_img(predicted_latents_rescaled)


    selected_indices = [14,18,9,10,4,5,2] 
    # Display samples from train dataset
    order = [0,2,1]
    
    # Save figure instead of displaying
    output_path = output_dir / "fig2_seed_to_intermediate.png"
    _save_predicted_images(y3[selected_indices,:,:,:], X[selected_indices,:,:,:], 
                          pred_images[selected_indices,:,:,:], len(selected_indices), 
                          output_path, order=order)
    
    return [output_path]