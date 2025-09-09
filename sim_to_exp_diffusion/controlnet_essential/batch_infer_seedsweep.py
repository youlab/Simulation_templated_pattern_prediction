import os
import cv2
import random
from pipeline import process
from datetime import datetime


currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year



# ─── Configuration ─────────────────────────────────────────────────────────────

# Path to the one image 
INPUT_DIR='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set/'
INPUT_IMAGE = os.path.join(INPUT_DIR, "196_1.TIF")  

# Output folder (will be created if needed)
OUTPUT_DIR = f"/hpc/dctrl/ks723/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_random_seed_sweep/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many random seeds to try
NUM_SEEDS = 100

# Fixed inference hyper-params (one sample only)
ARGS = {
    "prompt":            "",  # your prompt
    "a_prompt":          "",  # any extra positive prompt
    "n_prompt":          "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
    "num_samples":       1,
    "image_resolution":  256,
    "ddim_steps":        50,
    "guess_mode":        False,
    "strength":          1.0,
    "scale":             15.1,
    # ‘seed’ will be set per-loop
    "eta":               0.0,
}

# ─── Load & preprocess the single image ────────────────────────────────────────

# Read as BGR, convert to RGB
img_bgr = cv2.imread(INPUT_IMAGE)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read {INPUT_IMAGE}")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ─── Sweep 100 random seeds ────────────────────────────────────────────────────

# (Optionally fix the RNG so you get the same 100 seeds each run)
random.seed(42)
for _ in range(NUM_SEEDS):
    seed = random.randint(0, 2**31 - 1)
    ARGS["seed"] = seed

    # run exactly one sample
    outputs = process(img_rgb, **ARGS)
    out_img = outputs[0]

    # save with the seed in the filename
    base = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]
    fn = f"{base}_seed{seed}.png"
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out_bgr)

    print(f"[seed={seed}] saved → {fn}")
