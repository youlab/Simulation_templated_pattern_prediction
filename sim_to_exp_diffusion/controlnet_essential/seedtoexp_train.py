from share import *
import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import Subset, DataLoader
from torchvision.utils import make_grid, save_image
from seedtoexp_dataset import MyDataset
from cldm.logger_custom import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = 'control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# build a fixed mini-set once (K samples)
K = 5
fixed_ds = Subset(MyDataset(), list(range(K)))
fixed_dl = DataLoader(fixed_ds, batch_size=K, shuffle=False, num_workers=0)
fixed_batch = next(iter(fixed_dl))  # dict like {'jpg','hint','txt'}


# minimal callback
class FixedEval(pl.Callback):
    def __init__(self, batch, every=1000, outdir="eval_fixed", steps=30, eta=0.0, scale=9.0):
        self.b, self.every, self.outdir = batch, every, outdir
        self.steps, self.eta, self.scale = steps, eta, scale
        os.makedirs(outdir, exist_ok=True)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = trainer.global_step
        if gs == 0 or gs % self.every:  # log exactly at multiples
            return
        was_train = pl_module.training
        pl_module.eval()

        # move fixed batch to device
        b = {k: (v.to(pl_module.device) if hasattr(v, "to") else v) for k, v in self.b.items()}

        # use model’s existing visualizer
        imgs = pl_module.log_images(
            b, N=K, ddim_steps=self.steps, eta=self.eta,
            unconditional_guidance_scale=self.scale
        )  # dict[str] -> BCHW in [-1,1]

        for name, t in imgs.items():
            if t.ndim != 4: continue
            x = (t.clamp(-1,1) + 1) / 2.0
            grid = make_grid(x, nrow=K)  # one row of K images
            save_image(grid, os.path.join(self.outdir, f"{gs:07d}_{name}.png"))

        if was_train: pl_module.train()



# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


tb_logger  = TensorBoardLogger(save_dir="tb_logs", name="controlnet_seedtoexp")
csv_logger = CSVLogger(save_dir="csv_logs", name="controlnet_seedtoexp")

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
fixed_cb = FixedEval(fixed_batch, every=300, outdir="eval_fixed", steps=30, eta=0.0, scale=9.0)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, fixed_cb], max_epochs=5)


# Train!
trainer.fit(model, dataloader)
