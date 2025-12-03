import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset
from cldm.preprocess import preprocess_experimental_backgroundwhite,preprocess_simulation_graybackground
from cldm.config import BASE_FOLDER,SPECIFIC_FOLDER_SIM, SPECIFIC_FOLDER_EXP

base_folder=BASE_FOLDER  # place prompt_simtoexp.json at this location from hf datasets
base_folder_i=SPECIFIC_FOLDER_SIM
base_folder_o=SPECIFIC_FOLDER_EXP


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(os.path.join(base_folder,'prompt_simtoexp.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source_path = os.path.join(base_folder_i,source_filename)
        target_path = os.path.join(base_folder_o,target_filename)

        source = preprocess_simulation_graybackground(source_path)
        target = preprocess_experimental_backgroundwhite(target_path)


        # Converting to 3 channels for simulation images
        source = np.repeat(source[:, :, np.newaxis] , 3, axis=2) 

        # already resized to 256x256 for efficient downsampling by U-Net in steps of 8
    
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

