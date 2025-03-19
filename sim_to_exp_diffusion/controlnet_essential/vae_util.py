import torch
from diffusers import AutoencoderKL

def load_vae(device):
    """Load the VAE from a pretrained Stable Diffusion model."""
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae.to(device)
    return vae


def encode_img(input_img,vae):
    """
    Takes input a grayscale image and returns the scaled version sampled from the latent space 
    """
    input_img = input_img.repeat(3, 1, 1)
    
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling to make outputs from -1 to 1 
    return 0.18215 * latent.latent_dist.sample()



def decode_img(latents, vae):
    """
    Decode latent representations using the VAE decoder.
    Returns decoded images scaled between 0 and 1.
    """
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


