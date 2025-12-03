# import libs
import os
import sys
from pathlib import Path
import glob
import cv2
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set deterministic CUDA behavior BEFORE importing torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR / "sim_to_exp_diffusion" / "controlnet_essential"))
sys.path.insert(0, str(REPO_DIR))

import torch
import einops
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from cldm.cldm import ControlLDM
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
import config
from cldm.preprocess import preprocess_simulation_graybackground, grayfordisplay, preprocess_experimental_backgroundwhite
from config_automate import OUTPUT_DIR_SIMTOEXP, SIM_FOLDER_TEST, SEED_FOLDER_TEST, EXP_FOLDER_TEST, CKPT_PATH

# Model and sampler - initialized lazily when needed
_model = None
_ddim_sampler = None

def _get_model_and_sampler():
    """Lazy initialization of model and sampler. Requires CUDA."""
    global _model, _ddim_sampler
    
    if _model is not None:
        return _model, _ddim_sampler
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Figure 5 requires CUDA GPU to load the ControlNet model.\n"
            "Please run on a machine with GPU, or generate other figures: --figures 1 2 3 4"
        )
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    yaml_config = str(REPO_DIR / "sim_to_exp_diffusion" / "controlnet_essential" / "models" / "cldm_v15.yaml")
    config_yaml = OmegaConf.load(yaml_config)
    params = OmegaConf.to_container(config_yaml.model.params, resolve=True)
    _model = ControlLDM.load_from_checkpoint(CKPT_PATH, **params).cuda()
    _ddim_sampler = DDIMSampler(_model)
    
    return _model, _ddim_sampler


def _process_image(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    """Process a single image through ControlNet (replaces pipeline.process)."""
    model, ddim_sampler = _get_model_and_sampler()
    
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = np.random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results


def _run_inference(input_dir, output_dir, args_dict):
    """Run ControlNet inference to generate predictions from simulation images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Pick one file per numeric prefix
    prefix_map = {}
    for fp in sorted(glob.glob(os.path.join(input_dir, "*.TIF"))):
        prefix = os.path.basename(fp).split("_")[0]
        if prefix not in prefix_map:
            prefix_map[prefix] = fp
    
    # Inference over each prefix
    for prefix, fp in prefix_map.items():
        # process images with the simulation preprocessing step
        img = preprocess_simulation_graybackground(fp)
        
        # run the inference
        outs = _process_image(img, **args_dict)
        
        # save as prefix_1.png … prefix_5.png
        for i, out in enumerate(outs, start=1):
            fn = f"{prefix}_{i}.png"
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, fn), out_bgr)
        
        print(f"Done: {prefix}")


def generate_fig5(output_dir):
    """Generate Figure 5: Simulation to experiment translation with ControlNet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------
    # Step 1: Run inference to generate predictions
    # ------------------------------
    
    # Set up intermediate predictions folder
    predictions_dir = output_dir / "predictions_intermediate"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    INPUT_DIR = SIM_FOLDER_TEST  # test images folder
    
    # fixed hyper-params:
    ARGS = {
        "prompt": "",
        "a_prompt": "",
        "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
        "num_samples": 5,
        "image_resolution": 256,
        "ddim_steps": 50,
        "guess_mode": False,
        "strength": 1.0,
        "scale": 15.1,
        "seed": 729397049,
        "eta": 0.0
    }
    
    # Run inference
    print("Running ControlNet inference...")
    _run_inference(INPUT_DIR, str(predictions_dir), ARGS)
    
    # ------------------------------
    # Step 2: Create figure plot
    # ------------------------------



    # ——— CONFIG ———
    print("Creating figure plot...")
    sample_ids = [198, 196, 200, 227, 245]   # your conditions
    n_exp = 2                   # number of real-exp replicates
    n_pred = 2                  # number of model samples per condition
    
    folders = {
        "seed": SEED_FOLDER_TEST,
        "sim": SIM_FOLDER_TEST,
        "exp": EXP_FOLDER_TEST,
        "pred": str(predictions_dir)  # Use the predictions we just generated
    }
    
    # ——— GATHER PATHS ———
    seed_paths = {}
    for sid in sample_ids:
        m = glob.glob(f"{folders['seed']}/{sid}_*.png")
        if not m:
            raise FileNotFoundError(f"No seed for sample {sid}")
        seed_paths[sid] = m[0]
    
    sim_paths = {}
    for sid in sample_ids:
        sims = glob.glob(f"{folders['sim']}/{sid}_*.TIF")
        if not sims:
            raise FileNotFoundError(f"No sim for sample {sid}")
        sim_paths[sid] = sorted(
            sims,
            key=lambda f: int(re.search(rf"{sid}_(\d+)\.TIF", f).group(1))
        )[0]
    
    def sorted_reps(folder, sid, ext, n):
        files = glob.glob(f"{folder}/{sid}_*.{ext}")
        return sorted(
            files,
            key=lambda f: int(re.search(rf"{sid}_(\d+)\.{ext}", f).group(1))
        )[:n]
    
    def sorted_reps_pred(folder, sid, n):
        pattern = os.path.join(folder, f"{sid}_*")
        files = [f for f in glob.glob(pattern) if f.lower().endswith('.png')]
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0].split('_')[-1]))
        return files[:n]
    
    exp_paths = {sid: sorted_reps(folders['exp'], sid, 'TIF', n_exp) for sid in sample_ids}
    pred_paths = {sid: sorted_reps_pred(folders['pred'], sid, n_pred) for sid in sample_ids}
    
    # ——— PROCESS IMAGES WITH PREPROCESSING FUNCTIONS ———
    processed_images = {}
    
    for sid in sample_ids:
        processed_images[sid] = {}
        
        # Process seed image (use Image.open for seeds since they're already processed)
        seed_processed = grayfordisplay(seed_paths[sid], img_type='seed')
        processed_images[sid]['seed'] = seed_processed
        
        # Process simulation image with preprocess_simulation
        sim_processed = preprocess_simulation_graybackground(sim_paths[sid])
        processed_images[sid]['sim'] = sim_processed
        
        # Process experimental images with preprocess_experimental
        processed_images[sid]['exp'] = []
        for exp_path in exp_paths[sid]:
            exp_processed = preprocess_experimental_backgroundwhite(exp_path)
            processed_images[sid]['exp'].append(exp_processed)
        
        # Process prediction images (simple cv2 read with background conversion)
        processed_images[sid]['pred'] = []
        for pred_path in pred_paths[sid]:
            pred_processed = cv2.imread(pred_path)
            pred_processed = cv2.cvtColor(pred_processed, cv2.COLOR_BGR2RGB)
            # Apply thresholding and background conversion
            pred_processed = cv2.threshold(pred_processed, 10, 255, cv2.THRESH_TOZERO)[1]
            pred_processed[np.all(pred_processed == [0, 0, 0], axis=-1)] = [255, 255, 255]
            processed_images[sid]['pred'].append(pred_processed)
    
    # ——— PLOT PROCESSED IMAGES ———
    ncols = 2 + n_exp + n_pred
    fig, axes = plt.subplots(
        len(sample_ids), ncols,
        figsize=(ncols * 2, len(sample_ids) * 2),
        layout='constrained',
    )
    
    for i, sid in enumerate(sample_ids):
        # Seed (grayscale)
        axes[i, 0].imshow(processed_images[sid]['seed'], cmap='gray')
        axes[i, 0].axis('off')
        
        # Sim (grayscale)
        axes[i, 1].imshow(processed_images[sid]['sim'], cmap='gray')
        axes[i, 1].axis('off')
        
        # Experimental replicates (grayscale)
        for j in range(n_exp):
            if j < len(processed_images[sid]['exp']):
                axes[i, 2 + j].imshow(processed_images[sid]['exp'][j], cmap='gray')
                axes[i, 2 + j].axis('off')
        
        # Model predictions (color)
        for k in range(n_pred):
            if k < len(processed_images[sid]['pred']):
                axes[i, 2 + n_exp + k].imshow(processed_images[sid]['pred'][k])
                axes[i, 2 + n_exp + k].axis('off')
    
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, hspace=0.01, wspace=0.01)
    
    # Save the figure
    output_path = output_dir / "fig5.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Processed {len(sample_ids)} samples and saved figure to {output_path}")
    return [output_path]