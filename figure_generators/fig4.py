# import libs
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
import pickle
import json
import matplotlib.ticker as ticker

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "seed_to_sim1_sim2_deterministic"))
sys.path.insert(0, str(REPO_DIR))

from config_automate import FPATH
from models.vae import decode_img
from models.dilResNet import PDEArenaDilatedResNet
from config_automate import SIMULATION_TEST_OUTPUT_FOLDER_TDB_2, SIMULATION_TEST_OUTPUT_FOLDER_2
from config_automate import LATENT_OUTPUT_SAVED, MODEL_SAVE_LOCATION, LOGS_SAVE_LOCATION
from utils.display import display_predictions_multiple_samples
from utils.preprocess import preprocess_simulation_output_data


def _save_predictions_fig4a(input_images, ground_truths, pred_images_list, num_samples_list, output_path):
    """Save prediction comparisons for fig4a."""
    display_predictions_multiple_samples(input_images, ground_truths, pred_images_list, num_samples_list)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def _save_loss_plot_fig4b(variable1_values_sorted, best_test_losses_sorted, output_path):
    """Save loss plot for fig 4b."""
    fig, ax = plt.figure(figsize=(14, 6)), plt.gca()
    ax.set_aspect(0.5)
    
    plt.plot(variable1_values_sorted, best_test_losses_sorted, '-', linewidth=3, markersize=12, color='black', alpha=0.5)
    plt.plot(variable1_values_sorted, best_test_losses_sorted, 'o', linewidth=3, markersize=12, color='black', alpha=0.8)
    plt.xscale('log')
    if FPATH:
        plt.xticks(font=FPATH, fontsize=25)
        plt.yticks(font=FPATH, fontsize=25)
    else:
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    plt.ylim(0, 1)
    
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def generate_fig4ab(output_dir):
    """Generate Figure 4a: Intermediate to complex pattern prediction without data augmentation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    '''
    Loading test data for model testing
    '''
    
    path_output = SIMULATION_TEST_OUTPUT_FOLDER_TDB_2
    path_input = SIMULATION_TEST_OUTPUT_FOLDER_2
    
    ## add some of the images from the main dataset that hasn't been used to train the network
    start_index = 70000
    end_index = 70100
    output_data = preprocess_simulation_output_data(path_output, start_index, end_index)
    
    # input data
    input_data = preprocess_simulation_output_data(path_input, start_index, end_index)  # both are simulations, so preprocess accordingly
    
    X = output_data
    y = input_data
    
    # size of input data is 32x32 and output data is 256x256
    X = (np.array(X).reshape(-1, 1, 256, 256))  # /255.0  # last one is grayscale first minus one is all x
    y = (np.array(y).reshape(-1, 1, 256, 256))  # /255.0
    
    # normalizing images here to be bw 0 and 1
    X = X / 255.0
    y = y / 255.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Convert numpy arrays to torch tensors
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    
    y3 = y.repeat(1, 3, 1, 1)
    y4 = y.repeat(1, 4, 1, 1)
    
    # Example usage
    model = PDEArenaDilatedResNet(
        in_channels=4,               # Input channels
        out_channels=4,              # Output channels
        hidden_channels=64,          # Number of hidden channels
        num_blocks=18,               # Number of dilated blocks
        dilation_rates=[1, 2, 4, 8], # Dilation rates for multi-scale feature capture
        activation=nn.ReLU,          # Activation function
        norm=True
    )
    
    '''
    Define pre-trained SD VAE for decoding orginal predicted patterns from the latent predicted patterns
    dilResnet will predict latent representation of patterns
    '''
    
    # Load the pre-trained VAE model
    pickle_in = open(LATENT_OUTPUT_SAVED, "rb")
    yprime_in = pickle.load(pickle_in)
    
    yprime_in = yprime_in[start_index:end_index, :, :, :]
    yprime_in = torch.Tensor(yprime_in)
    
    '''
    Displaying predictions for indices 5,6,10 with a specific number of seeds(to serve as generalized illustration across whole test set)
    '''


    
    pattern = r"Pixel_32x32x4to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_([\d]+)_v1015-([\d]+)_best_tolerant.*\.pt"
    
    def extract_number(filename):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))  # Extract the number of images
        return float('inf')
    
    # Directory containing the model files (WITHOUT augmentation for fig4a)
    directory = os.path.join(MODEL_SAVE_LOCATION, "models_Fig4")
    
    # List all files and filter out those matching the pattern
    filtered_filenames = [filename for filename in os.listdir(directory) if re.match(pattern, filename)]
    
    # Sort the filenames based on the number of images used in training
    sorted_filenames = sorted(filtered_filenames, key=extract_number)
    # Initialize lists to collect predictions and number of samples
    pred_images_list = []
    num_samples_list = []
    
    sample_indices = [6, 10, 5]
    
    # Get the input images and ground truths for the fixed samples
    input_images = y3[sample_indices].cpu()  # Ensure tensors are on CPU
    ground_truths = X[sample_indices].cpu()
    
    # Loop through each model file
    for filename in sorted_filenames:
        num_samples = extract_number(filename)
        num_samples_list.append(num_samples)
        
        print(f"Processing model trained with {num_samples} samples: {filename}")
        
        # Load the model
        model.load_state_dict(torch.load(os.path.join(directory, filename), map_location=device))
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Make predictions for the selected input images
            predicted_latents = model(yprime_in[sample_indices].to(device))
            # Decode the predictions
            pred_images = decode_img(predicted_latents)
            pred_images_list.append(pred_images.cpu())  # Move to CPU for plotting
    
    # Save the predictions figure
    output_path_predictions = output_dir / "fig4a_predictions.png"
    _save_predictions_fig4a(input_images, ground_truths, pred_images_list, num_samples_list, output_path_predictions)
    
    # Directory location where your json files are stored
    directory_location = LOGS_SAVE_LOCATION
    
    # Define a regex pattern to extract variable1 from the filenames (WITHOUT augmentation)
    pattern = re.compile(r"losses_Pixel_32x32x4to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_(\d+)_v1015-")
    
    # Initialize lists to store variable1 and the corresponding last test loss
    variable1_values = []
    best_test_losses = []
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory_location):
        if filename.endswith(".json"):
            match = pattern.match(filename)
            if match:
                variable1 = int(match.group(1))
                with open(os.path.join(directory_location, filename), 'r') as file:
                    data = json.load(file)
                    # get the best test loss value
                    # best_test_loss = data["test_losses"][-1]
                    best_test_loss = data["tolerant_loss"]
                    variable1_values.append(variable1)
                    best_test_losses.append(best_test_loss)
    
    # Sort the values based on variable1 for better visualization
    sorted_data = sorted(zip(variable1_values, best_test_losses))
    variable1_values_sorted, best_test_losses_sorted = zip(*sorted_data)
    
    # Save the loss plot
    output_path_loss = output_dir / "fig4b_loss.png"
    _save_loss_plot_fig4b(variable1_values_sorted, best_test_losses_sorted, output_path_loss)
    
    return [output_path_predictions, output_path_loss]





def _save_predictions_fig4c(input_images, ground_truths, pred_images_list, num_samples_list, output_path):
    """Save prediction comparisons for fig4c"""
    display_predictions_multiple_samples(input_images, ground_truths, pred_images_list, num_samples_list)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def _save_loss_plot_fig4d(variable1_values_sorted, best_test_losses_sorted, output_path):
    """Save loss plot for fig4d"""
    fig, ax = plt.figure(figsize=(14, 6)), plt.gca()
    ax.set_aspect(0.5)
    
    plt.plot(variable1_values_sorted, best_test_losses_sorted, '-', linewidth=3, markersize=12, color='black', alpha=0.5)
    plt.plot(variable1_values_sorted, best_test_losses_sorted, 'o', linewidth=3, markersize=12, color='black', alpha=0.8)
    plt.xscale('log')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylim(0, 1)
    
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.yaxis.set_major_locator(ticker.LinearLocator(3))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def generate_fig4cd(output_dir):
    """Generate Figure 4b: Intermediate to complex pattern prediction with data augmentation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    '''
    Loading test data for model testing
    '''
    
    # prediction from testing dataset
    
    path_output = SIMULATION_TEST_OUTPUT_FOLDER_TDB_2
    path_input = SIMULATION_TEST_OUTPUT_FOLDER_2
    
    ## add some of the images from the main dataset that hasn't been used to train the network
    start_index = 70000
    end_index = 70100
    output_data = preprocess_simulation_output_data(path_output, start_index, end_index)
    
    # input data
    input_data = preprocess_simulation_output_data(path_input, start_index, end_index)  # both are simulations, so preprocess accordingly
    
    X = output_data
    y = input_data
    
    # size of input data is 32x32 and output data is 256x256
    X = (np.array(X).reshape(-1, 1, 256, 256))  # /255.0  # last one is grayscale first minus one is all x
    y = (np.array(y).reshape(-1, 1, 256, 256))  # /255.0
    
    # normalizing images here to be bw 0 and 1
    X = X / 255.0
    y = y / 255.0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Convert numpy arrays to torch tensors
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    
    y3 = y.repeat(1, 3, 1, 1)
    y4 = y.repeat(1, 4, 1, 1)
    
    '''
    Defining dilResNet model for loading model weights
    '''
    
    # Example usage
    model = PDEArenaDilatedResNet(
        in_channels=4,               # Input channels
        out_channels=4,              # Output channels
        hidden_channels=64,          # Number of hidden channels
        num_blocks=18,               # Number of dilated blocks
        dilation_rates=[1, 2, 4, 8], # Dilation rates for multi-scale feature capture
        activation=nn.ReLU,          # Activation function
        norm=True
    )
    
    '''
    Define pre-trained SD VAE for decoding orginal predicted patterns from the latent predicted patterns
    dilResnet will predict latent representation of patterns
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    pickle_in = open(LATENT_OUTPUT_SAVED, "rb")
    yprime_in = pickle.load(pickle_in)
    
    yprime_in = yprime_in[start_index:end_index, :, :, :]
    yprime_in = torch.Tensor(yprime_in)
    
    pattern = r"dilRESNET_DataAugmentation_intermediatetocomplex_([\d]+)_v1015-([\d]+)_best_tolerant.*\.pt"
    
    def extract_number(filename):
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))  # Extract the number of images
        return float('inf')
    
    # Directory containing the model files (WITH augmentation for fig4c)
    directory = os.path.join(MODEL_SAVE_LOCATION, "models_dataaugmentation_Fig4")
    
    # List all files and filter out those matching the pattern
    filtered_filenames = [filename for filename in os.listdir(directory)
                         if re.match(pattern, filename) and filename.endswith('_best_tolerant.pt')]
    
    # Sort the filenames based on the number of images used in training
    sorted_filenames = sorted(filtered_filenames, key=extract_number)
    
    '''
    Displaying predictions for indices 5,6,10 with a specific number of seeds(to serve as generalized illustration across whole test set)
    '''
    # Initialize lists to collect predictions and number of samples
    pred_images_list = []
    num_samples_list = []
    
    sample_indices = [6, 10, 5]
    
    # Get the input images and ground truths for the fixed samples
    input_images = y3[sample_indices].cpu()  # Ensure tensors are on CPU
    ground_truths = X[sample_indices].cpu()
    
    # Loop through each model file
    for filename in sorted_filenames:
        num_samples = extract_number(filename)
        num_samples_list.append(num_samples)
        
        print(f"Processing model trained with {num_samples} samples: {filename}")
        
        # Load the model
        model.load_state_dict(torch.load(os.path.join(directory, filename), map_location=device))
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Make predictions for the selected input images
            predicted_latents = model(yprime_in[sample_indices].to(device))
            # Decode the predictions
            pred_images = decode_img(predicted_latents)
            pred_images_list.append(pred_images.cpu())  # Move to CPU for plotting
    
    # Save the predictions figure (skip index 3)
    output_path_predictions = output_dir / "fig4c_predictions.png"
    _save_predictions_fig4c(input_images, ground_truths, pred_images_list[0:3] + pred_images_list[4:], num_samples_list, output_path_predictions)
    
    # Directory location where your json files are stored
    directory_location = LOGS_SAVE_LOCATION
    
    # Define a regex pattern to extract variable1 from the filenames (WITH augmentation)
    pattern = re.compile(r"losses_dilRESNET_DataAugmentation_intermediatetocomplex_(\d+)_v1015-")
    
    # Initialize lists to store variable1 and the corresponding last test loss
    variable1_values = []
    best_test_losses = []
    
    # Iterate through the files in the directory
    for filename in os.listdir(directory_location):
        if filename.endswith(".json"):
            match = pattern.match(filename)
            if match:
                variable1 = int(match.group(1))
                with open(os.path.join(directory_location, filename), 'r') as file:
                    data = json.load(file)
                    # get the best test loss value
                    # best_test_loss = data["test_losses"][-1]
                    best_test_loss = data["tolerant_loss"]
                    variable1_values.append(variable1)
                    best_test_losses.append(best_test_loss)
    
    # Sort the values based on variable1 for better visualization
    sorted_data = sorted(zip(variable1_values, best_test_losses))
    variable1_values_sorted, best_test_losses_sorted = zip(*sorted_data)
    
    # remove the 4th value, index 3 for plotting purposes
    variable1_values_sorted = list(variable1_values_sorted)
    best_test_losses_sorted = list(best_test_losses_sorted)
    del variable1_values_sorted[3]
    del best_test_losses_sorted[3]
    
    # Save the loss plot
    output_path_loss = output_dir / "fig4d_loss.png"
    _save_loss_plot_fig4d(variable1_values_sorted, best_test_losses_sorted, output_path_loss)
    
    return [output_path_predictions, output_path_loss]