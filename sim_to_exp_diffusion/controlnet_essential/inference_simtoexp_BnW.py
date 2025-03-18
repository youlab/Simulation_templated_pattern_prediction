import os
import cv2
import numpy as np
import torch
import random
import einops
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from datetime import datetime

# Import model and sampler
from cldm.cldm import ControlLDM  # your model class
from cldm.ddim_hacked import DDIMSampler
from dataset_test import TestDataset  # your dataset class
from annotator.util import resize_image, HWC3
import config



# ------------------------------
# Set up output folder using current date/time
# ------------------------------
currentSecond = datetime.now().second
currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year

task_inference='simtoexp_BnW'

output_folder = f"/hpc/dctrl/ks723/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_{task_inference}"
os.makedirs(output_folder, exist_ok=True)



# ------------------------------
# Parameters and File Paths (update these as needed)
# ------------------------------
test_folder     = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set"  # test images folder
ckpt_path       = "/hpc/dctrl/ks723/Huggingface_repos/ControlNet_repo/controlnet_repo/lightning_logs/version_25484631/checkpoints/epoch=4-step=51124.ckpt"  # your ckpt file
yaml_config     = "./models/cldm_v15.yaml"  # YAML config file

# Inference parameters
prompt          = ""        # blank prompt (or set as desired)
a_prompt        = ""        # additional prompt (if any)
n_prompt        = ""        # negative prompt (if any)
num_samples     = 3         # generate multiple samples per image
image_resolution= 256       # resolution for resizing input image
ddim_steps      = 50       # number of DDIM sampling steps
scale           = 9.0       # guidance scale
seed_value      = 729397049        # -1 means random seed
eta             = 0.0       # DDIM eta
guess_mode      = True     # set to True if you want guess mode behavior
strength        = 1.0       # control strength used in magic scaling

# ------------------------------
# 1. Load the Model from Checkpoint
# ------------------------------
config_yaml = OmegaConf.load(yaml_config)
params = OmegaConf.to_container(config_yaml.model.params, resolve=True)
model = ControlLDM.load_from_checkpoint(ckpt_path, **params)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# ------------------------------
# 2. Prepare the Test Dataset and DataLoader
# ------------------------------
test_dataset = TestDataset(test_folder, preprocess_mode="simulation", prompt_file=None)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# ------------------------------
# 3. Inference Loop for Multiple Samples per Image
# ------------------------------
for i, (batch, img_path) in enumerate(dataloader):
    with torch.no_grad():
        # Preprocess the input image (similar to process() in Gradio)
        # Use the "hint" field from the dataset as the input image.

        ###############################

        # (Assumes TestDataset returns a dict with keys: "jpg", "txt", "hint")
        input_img = batch["hint"]  # expected to be a numpy array, but is not- torch tensor

        print(type(input_img))
        
        if isinstance(input_img, torch.Tensor):
            # input_img is expected to be in (C, H, W) and in float [0,1]
            input_img = input_img.cpu().numpy()         # shape: (C, H, W)
            # print(input_img.shape)

            if input_img.ndim == 4 and input_img.shape[0] == 1:
                input_img = input_img[0]

            # input_img = np.transpose(input_img, (1, 2, 0))  # shape: (H, W, C)
            input_img = (input_img * 255).astype(np.uint8)  # convert to uint8  # commenting this 7.56pm 02/11

        img = resize_image(HWC3(input_img), image_resolution)
        H, W, C = img.shape


        ###################################

        # img = resize_image(HWC3(batch["hint"]), image_resolution)
        # H, W, _ = img.shape






        # Use the original image as the control input.
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        # Replicate the control image for each sample.
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Set seed for reproducibility.
        if seed_value == -1:
            seed_value = random.randint(0, 65535)
        seed_everything(seed_value)

        if hasattr(config, "save_memory") and config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Build conditioning dictionary.
        # For text conditioning, concatenate prompt and a_prompt and repeat for num_samples.
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        # Determine latent shape from the resized image.
        shape = (4, H // 8, W // 8)

        if hasattr(config, "save_memory") and config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Include magic scaling for control scales.
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode else ([strength] * 13)
        )

        # Run DDIM sampling.
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if hasattr(config, "save_memory") and config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode the latent samples into images.
        x_samples = model.decode_first_stage(samples)
        # x_samples = einops.rearrange(x_samples, "b c h w -> b h w c")
        # x_samples = (x_samples * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        # Save the generated images.
        for j in range(num_samples):
            base_name = os.path.splitext(os.path.basename(img_path[0]))[0]
            out_filename = f"{base_name}_{j}.png" if num_samples > 1 else f"{base_name}.png"
            out_path = os.path.join(output_folder, out_filename)
            cv2.imwrite(out_path, cv2.cvtColor(results[j], cv2.COLOR_RGB2BGR))
            print(f"Saved {out_path}")

print("Inference complete. Predictions saved in", output_folder)

