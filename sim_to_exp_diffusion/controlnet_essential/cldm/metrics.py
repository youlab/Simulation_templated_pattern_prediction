from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from cldm.config import FPATH
import torch 
import lpips
import cv2
import torch.nn.functional as F

"""Setup for LPIPS, SSIM and ORB calculations in batches"""

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='vgg').to(device).eval()
orb         = cv2.ORB_create()
bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def compute_ssim_high_precision(original_images, reconstructed_images):
    ssim_values = []
    batch_size = original_images.shape[0]

    for i in range(batch_size):
        # Convert tensors to numpy arrays
        original = original_images[i].cpu().numpy()  # Shape: [channels, height, width]
        reconstructed = reconstructed_images[i].cpu().numpy()

        # Ensure images are in the range [0, 1]
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)

        # Convert to [height, width, channels]
        original = np.transpose(original, (1, 2, 0))  # Shape: [height, width, channels]
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

        # Convert reconstructed RGB image to grayscale using standard luminance formula
        if reconstructed.shape[2] == 3:
            reconstructed_gray = np.dot(reconstructed[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            reconstructed_gray = reconstructed.squeeze(axis=2)

        # Ensure original image is grayscale
        if original.shape[2] == 1:
            original_gray = original.squeeze(axis=2)
        else:
            original_gray = np.dot(original[..., :3], [0.2989, 0.5870, 0.1140])

        # Compute SSIM
        ssim_index = ssim(original_gray, reconstructed_gray, data_range=1.0)
        ssim_values.append(ssim_index)

    return ssim_values


def display_images_with_ssim(reference_images, comparison_images,num_samples=3):
    # Compute SSIM values using high-precision data
    ssim_values = compute_ssim_high_precision(reference_images, comparison_images)

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 4, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.5)

    for i in range(num_samples):
        # Get the reference and comparison images
        ref_image = reference_images[i].cpu().numpy()  # Shape: [C, H, W] or [H, W]
        comp_image = comparison_images[i].cpu().numpy()  # Shape: [C, H, W]

        print(f"Sample {i}: ref_image.shape = {ref_image.shape}, comp_image.shape = {comp_image.shape}")

        # Handle the reference image (ground truth)
        if ref_image.ndim == 3 and ref_image.shape[0] == 1:
            # Grayscale image, squeeze the channel dimension
            ref_image_disp = ref_image.squeeze(0)  # Shape: [H, W]
        elif ref_image.ndim == 3:
            # If the image has more channels, convert to grayscale
            ref_image_disp = np.transpose(ref_image, (1, 2, 0))
            ref_image_disp = np.dot(ref_image_disp[..., :3], [0.2989, 0.5870, 0.1140])
        elif ref_image.ndim == 2:
            # Already a 2D grayscale image
            ref_image_disp = ref_image  # Shape: [H, W]
        else:
            raise ValueError(f"Unexpected ref_image shape: {ref_image.shape}")

        # Handle the comparison image (predicted image)
        if comp_image.ndim == 3 and comp_image.shape[0] == 3:
            # Transpose to [H, W, C]
            comp_image_disp = np.transpose(comp_image, (1, 2, 0))
            # Convert RGB to grayscale
            comp_image_disp = np.dot(comp_image_disp[..., :3], [0.2989, 0.5870, 0.1140])
        elif comp_image.ndim == 3 and comp_image.shape[0] == 1:
            comp_image_disp = comp_image.squeeze(0)  # Shape: [H, W]
        elif comp_image.ndim == 2:
            comp_image_disp = comp_image  # Shape: [H, W]
        else:
            raise ValueError(f"Unexpected comp_image shape: {comp_image.shape}")

        
    
        # Display the reference image
        axes[0, i].imshow(ref_image_disp, cmap='gray')
        axes[0, i].axis('off')
        # axes[0, i].set_title('Ground Truth')

        # Display the comparison image
        axes[1, i].imshow(comp_image_disp, cmap='gray')
        axes[1, i].axis('off')
        # axes[1, i].set_title('Predicted Image')

        # Display the SSIM score
        axes[2, i].text(0.5, 0.5, f'{ssim_values[i]:.3f}',
                        ha='center', va='center', fontsize=50,font=FPATH)
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()




def calculate_lpips_score_batch(imgs1: torch.Tensor, imgs2: torch.Tensor) -> np.ndarray:
    """
    imgs1, imgs2: [B, C, H, W] in [0,1]
    returns: np.ndarray of shape [B] with LPIPS scores
    """
    if imgs1 is None or imgs2 is None:
        return np.full((0,), np.nan)

    scores = []
    for img1, img2 in zip(imgs1, imgs2):
        # bring to [1,C,H,W] on correct device
        x1 = img1.unsqueeze(0).to(device).float()
        x2 = img2.unsqueeze(0).to(device).float()

        # if single‐channel, repeat to 3
        if x1.size(1) == 1:
            x1 = x1.repeat(1, 3, 1, 1)
            x2 = x2.repeat(1, 3, 1, 1)

        # resize exactly as your PIL‐Resize((256,256))
        x1 = F.interpolate(x1, size=(256, 256), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x2, size=(256, 256), mode='bilinear', align_corners=False)

        # normalize to [-1,1] if currently in [0,1]
        if x1.min() >= 0 and x1.max() <= 1:
            x1 = x1 * 2 - 1
            x2 = x2 * 2 - 1

        with torch.no_grad():
            score = lpips_model(x1, x2).item()
        scores.append(score)

    return np.array(scores)


def calculate_orb_similarity_batch(imgs1: torch.Tensor, imgs2: torch.Tensor) -> np.ndarray:
    """
    imgs1, imgs2: [B, C, H, W] in [0,1]
    returns: np.ndarray of shape [B] with ORB match‐fraction scores
    """
    if imgs1 is None or imgs2 is None:
        return np.full((0,), np.nan)

    scores = []
    for img1, img2 in zip(imgs1, imgs2):
        # to H×W×C uint8
        arr1 = (img1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        arr2 = (img2.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # convert to gray exactly as PIL.convert('L')
        gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY) if arr1.ndim == 3 else arr1
        gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY) if arr2.ndim == 3 else arr2

        # match shapes
        if gray2.shape != gray1.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None or not kp1 or not kp2:
            scores.append(0.0)
        else:
            matches = bf.match(des1, des2)
            scores.append(len(matches) / max(len(kp1), len(kp2)))

    return np.array(scores)


def calculate_ssim_batch(original_images: torch.Tensor,
                       reconstructed_images: torch.Tensor) -> np.ndarray:
    """
    original_images, reconstructed_images: [B, C, H, W] in [0,1]
    returns: np.ndarray of shape [B] with SSIM scores
    """
    B, C, H, W = original_images.shape
    scores = []

    for i in range(B):
        orig = original_images[i].cpu().numpy()       # [C, H, W]
        recon = reconstructed_images[i].cpu().numpy()

        # clip to [0,1]
        orig = np.clip(orig, 0, 1)
        recon = np.clip(recon, 0, 1)

        # to [H, W, C]
        orig = np.transpose(orig, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))

        # RGB → gray
        if orig.shape[2] == 3:
            orig_gray = np.dot(orig[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            orig_gray = orig.squeeze(axis=2)

        if recon.shape[2] == 3:
            recon_gray = np.dot(recon[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            recon_gray = recon.squeeze(axis=2)

        score = ssim(orig_gray, recon_gray, data_range=1.0)
        scores.append(score)

    return np.array(scores)



