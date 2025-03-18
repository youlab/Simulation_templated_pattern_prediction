import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset
from cldm.preprocess import preprocess_simulation,preprocess_experimental

base_folder='/hpc/group/youlab/ks723/storage'
base_folder_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_100AUG'
base_folder_o='/hpc/group/youlab/ks723/storage/Exp_images/Final_folder_uniform_fixedseed_100AUG'


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

        source_path= os.path.join(base_folder_i,source_filename)
        target_path= os.path.join(base_folder_o,target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = preprocess_simulation(source_path)
        target = preprocess_experimental(target_path)


        # Repeat across 3 channels to make it (H, W, 3)
        source = np.repeat(source[:, :, np.newaxis] , 3, axis=2) 
        target= np.repeat(target[:, :, np.newaxis] , 3, axis=2) 
        
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


