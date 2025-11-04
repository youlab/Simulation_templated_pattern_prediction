import os
import cv2
import random
from pipeline import process
from datetime import datetime
from cldm.preprocess import preprocess_simulation_graybackground


currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year



# ─── Configuration ─────────────────────────────────────────────────────────────

# Path to the one image 
INPUT_DIR='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set/'
INPUT_IMAGE_PATH = os.path.join(INPUT_DIR, "196_1.TIF")  
# preprocess the input image to get numpy array
preprocessed_array = preprocess_simulation_graybackground(INPUT_IMAGE_PATH)
if preprocessed_array is None:
    raise FileNotFoundError(f"Failed to preprocess {INPUT_IMAGE_PATH}")





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

# The preprocessed_array is already a grayscale numpy array, convert to RGB for the pipeline
if len(preprocessed_array.shape) == 2:
    # Grayscale image, convert to RGB
    img_rgb = cv2.cvtColor(preprocessed_array, cv2.COLOR_GRAY2RGB)
else:
    # Already has channels
    img_rgb = preprocessed_array

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
    base = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    fn = f"{base}_seed{seed}.png"
    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out_bgr)

    print(f"[seed={seed}] saved → {fn}")
