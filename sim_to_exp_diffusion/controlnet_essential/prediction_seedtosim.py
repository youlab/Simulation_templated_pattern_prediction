import torch
from models import PDEArenaDilatedResNet
from vae_util import load_vae
from predict_util import predict_from_seed

# Define the model name and path to the checkpoint.
MODEL_NAME = 'Pixel_32x32x3to32x32x4_dilRESNET_30k_newpatterns_seedtointermediate__Model_v1113_Cluster_GPU_tfData-1731542355'
MODEL_PATH = f'/hpc/group/youlab/ks723/miniconda3/saved_models/trained/{MODEL_NAME}.pt'

def prediction(seed_image):
    """
    High-level prediction function to be called from the Gradio interface.
    Given a 3-channel seed image (numpy array), it returns the predicted image tensor.
    
    Usage:
        pred = prediction(seed_image)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load the VAE.
    vae = load_vae(device)
    
    # Instantiate and load the ResNet model.
    model = PDEArenaDilatedResNet(in_channels=3, out_channels=4, hidden_channels=64, num_blocks=15, dilation_rates=[1, 2, 4, 8])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    # Predict and decode the latent representation.
    predicted_images = predict_from_seed(seed_image, model, vae, device)
    return predicted_images
