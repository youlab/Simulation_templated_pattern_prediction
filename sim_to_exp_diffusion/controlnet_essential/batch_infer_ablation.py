import os
import glob
import cv2
import argparse
import pipeline     
from datetime import datetime
from cldm.preprocess import preprocess_simulation_graybackground
import argparse

p = argparse.ArgumentParser()
p.add_argument('--specific_folder', type=str, default='simtoexp')

currentMinute = datetime.now().minute
currentHour   = datetime.now().hour
currentDay    = datetime.now().day
currentMonth  = datetime.now().month
currentYear   = datetime.now().year

p.add_argument("--input-dir",
                default="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Final_Test_set",
                help="Folder of test images")
p.add_argument("--task", required=True,
                help="specific folder suffix to save results")
p.add_argument("--seed",          type=int,   default=729397049)
p.add_argument("--num-samples",   type=int,   default=5)
p.add_argument("--image-resolution", type=int, default=256)
p.add_argument("--ddim-steps",    type=int,   default=50)
p.add_argument(
    "--guess-mode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable or disable guess mode"
    )
p.add_argument("--strength",      type=float, default=1.0)
p.add_argument("--scale",         type=float, default=15.1)
p.add_argument("--eta",           type=float, default=0.0)
p.add_argument("--a-prompt",      type=str,   default="")
p.add_argument("--n-prompt",      type=str,
                default="longbody, lowres, bad anatomy, cropped, worst quality, low quality")
args = p.parse_args()

model, sampler = pipeline.build(args.specific_folder)

# inference base folder
output_dir=f"/hpc/dctrl/ks723/Physics_constrained_DL_pattern_prediction/sim_to_exp_diffusion/controlnet_essential/inference/v{currentYear}{currentMonth}{currentDay}_{currentHour}{currentMinute}_{args.task}"
os.makedirs(output_dir, exist_ok=True)

ARGS = {
    "prompt":           "",
    "a_prompt":         args.a_prompt,
    "n_prompt":         args.n_prompt,
    "num_samples":      args.num_samples,
    "image_resolution": args.image_resolution,
    "ddim_steps":       args.ddim_steps,
    "guess_mode":       args.guess_mode,
    "strength":         args.strength,
    "scale":            args.scale,
    "seed":             args.seed,
    "eta":              args.eta,
}

# pick one file per prefix
prefix_map = {}
for fp in sorted(glob.glob(os.path.join(args.input_dir, "*.TIF"))):
    prefix = os.path.basename(fp).split("_")[0]
    if prefix not in prefix_map:
        prefix_map[prefix] = fp

# inference loop
for prefix, fp in prefix_map.items():

    img = preprocess_simulation_graybackground(fp)
    outs = pipeline.process(model, sampler, img, **ARGS)
    for i, out in enumerate(outs, start=1):
        fn = f"{prefix}_{i}.png"
        cv2.imwrite(
            os.path.join(output_dir, fn),
            cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        )
    print("Done:", prefix)


