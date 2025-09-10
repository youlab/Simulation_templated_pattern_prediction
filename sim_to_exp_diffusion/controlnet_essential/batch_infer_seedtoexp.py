import os, glob, cv2
from pipeline_seedtoexp import process
from datetime import datetime
import time

# ------------------------------
# Set up output folder using current date/time
# ------------------------------

currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year

task_inference='seedtoexp'

OUTPUT_DIR = f"/hpc/dctrl/ks723/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_{task_inference}"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ------------------------------
# Parameters and File Paths (update these as needed)
# ------------------------------
INPUT_DIR      = "/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set_input"  # test images folder


os.makedirs(OUTPUT_DIR, exist_ok=True)

# fixed hyper-params:
ARGS = {
  "prompt":   "",
  "a_prompt": "",
  "n_prompt": "longbody, lowres, bad anatomy, cropped, worst quality, low quality",
  "num_samples":   5,
  "image_resolution":256,
  "ddim_steps":     50,
  "guess_mode":     False,
  "strength":       1.0,
  "scale":          15.1,
  "seed":           729397049,
  "eta":            0.0
}

# ------------------------------
# Pick one file per numeric prefix
# ------------------------------
prefix_map = {}
for fp in sorted(glob.glob(os.path.join(INPUT_DIR, "*.png"))):
    prefix = os.path.basename(fp).split("_")[0]
    if prefix not in prefix_map:
        prefix_map[prefix] = fp




# ------------------------------
# Inference over each prefix
# ------------------------------
for prefix, fp in prefix_map.items():
    # load & convert to H×W×C RGB
    img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    img=  cv2.resize(img, (256, 256))  # resize to match model input size

    # run the shared pipeline
    outs = process(img, **ARGS)

    # save as prefix_1.png … prefix_5.png
    for i, out in enumerate(outs, start=1):
        fn  = f"{prefix}_{i}.png"
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, fn), out_bgr)

    print(f"Done: {prefix}")