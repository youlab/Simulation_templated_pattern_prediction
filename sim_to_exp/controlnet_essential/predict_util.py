import torch
import numpy as np

def predict_from_seed(seed_image, model, vae, device):
    """
    Given a 3-channel seed image (as a numpy array),
    run the trained ResNet model to obtain latent predictions,
    then decode them using the VAE.
    """
    # Convert the seed image (H, W, 3) to a tensor normalized in [0,1] and shape (1, 3, H, W)
    seed_tensor = torch.from_numpy(seed_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_latents = model(seed_tensor)
    
    # Import decode_img locally to avoid circular imports
    from vae_util import decode_img
    predicted_images = decode_img(predicted_latents, vae)
    return predicted_images
