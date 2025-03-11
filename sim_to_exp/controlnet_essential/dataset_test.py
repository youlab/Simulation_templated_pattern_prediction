import json
import cv2
import numpy as np
import os
import torch 
from torch.utils.data import Dataset
from cldm.preprocess import preprocess_simulation,preprocess_experimental,preprocess_seed

# right now the test dataset accepts the image folder as argument 

# preprocessing info - experimental, expBnW, or simulation 


class TestDataset(Dataset):
    def __init__(self, image_folder,preprocess_mode="simulation", prompt_file=None):
        self.image_paths = sorted(
            [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg",".TIF"))]
        )
        self.preprocess_mode=preprocess_mode

        # Load prompts if available
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                self.prompts = json.load(f)
        else:
            self.prompts = {os.path.basename(path): "" for path in self.image_paths}  # Default to empty prompts

        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
       
        if self.preprocess_mode=="simulation":
            image = preprocess_simulation(img_path)  # Apply the same preprocessing as training

        elif self.preprocess_mode=='experimental':
            image=preprocess_experimental(img_path)  

        elif self.preprocess_mode=='seed':
            image=preprocess_seed(img_path)

        else:
            raise ValueError(f"Unknown preprocess_mode: {self.preprocess_mode}")
        


        # Ensure grayscale images are triplicated to (H, W, 3)
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Get corresponding text prompt (or use empty string if missing)
        prompt = self.prompts.get(os.path.basename(img_path), "")

        dummy_target = np.full(image.shape, -1.0, dtype=np.float32)

        return dict(jpg=dummy_target, txt=prompt,hint=image), img_path


