# import libraries 

import os
from pathlib import Path
import torch
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
import cv2
# import time 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle 





rfactor=256   # this is for resizing the final patterns #changed 070924
# rfactor=128   # changed on 050324

img_length=rfactor
img_width=rfactor

# # for local Windows CPU 
# datadir_i="C:/Users/kinsh/Downloads/Test_test2final/Sim_input_3"
# datadir_o="C:/Users/kinsh/Downloads/Test_test2final/Sim_output_3"

# for CentOS 8 cluster 
# datadir_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL'
# datadir_o="/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_071223/Sim_output"

# datadir_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_plusaug'

# datadir_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_aug4' #changed 041224 test set
datadir_i='/hpc/group/youlab/ks723/storage/MATLAB_SIMS/Sim_031524/Selected_v4_ALL_100AUG' #changed 042524



path_i=os.path.join(datadir_i)
# path_o=os.path.join(datadir_o)

training_data=[]
# test_data=[]

# parameters for image

img_shape_i=256 # to make image 256x256 after cropping the image  #changed on 070924  
# img_shape_i=128 # # changed on 050324
# img_shape_o=32   # to make image 32x32 after cropping the image 


# added code for cropping
# assuming same shape for each image, image not normalized before cropping 
# change the heigth and width option later outside the for loop for faster processing 


# parameters for cropping output of sim
top_crop_i = 30
bottom_crop_i = 30
left_crop_i = 31
right_crop_i = 30


####################

### MODIFIED 070924
# Implemented sorting to sort files in proper order 

sorted_filenames_path = '/hpc/group/youlab/ks723/miniconda3/Lingchong/sorted_files_full.txt' 

def load_sorted_filenames(file_path):
    with open(file_path, 'r') as file:
        sorted_filenames = [line.strip() for line in file.readlines()]
    return sorted_filenames

##################





def create_training_data(top_crop,bottom_crop,left_crop,right_crop):
    # img_filenames_i = sorted(os.listdir(path_i))
    img_filenames_i=load_sorted_filenames(sorted_filenames_path)   ##### MODIFIED ON 070924

    for img in img_filenames_i:

        img_array_i=cv2.imread(os.path.join(path_i,img),cv2.IMREAD_GRAYSCALE)  # ,cv2.IMREAD_GRAYSCALE removed on 050324

        new_height = img_array_i.shape[0] - (top_crop + bottom_crop)
        new_width = img_array_i.shape[1] - (left_crop + right_crop)
    
        new_array_i = img_array_i[top_crop:top_crop+new_height, left_crop:left_crop+new_width]


        # blurred = cv2.GaussianBlur(img_array_i, (7, 7), 0)
        # (T, img_array_i) = cv2.threshold(blurred, 200, 255,cv2.THRESH_BINARY)   # removing _INV
        new_array_i=cv2.resize(new_array_i,(img_length,img_width))
        

        training_data.append([new_array_i])


create_training_data(top_crop_i,bottom_crop_i,left_crop_i,right_crop_i)


# # parameters for croppping input seed

# top_crop_o=0
# bottom_crop_o=0
# left_crop_o=3
# right_crop_o=2

# def create_test_data(top_crop,bottom_crop,left_crop,right_crop):
#     img_filenames_o = sorted(os.listdir(path_o))
#     for img in img_filenames_o:
#         img_array_o=cv2.imread(os.path.join(path_o,img),cv2.IMREAD_GRAYSCALE)

#         new_height = img_array_o.shape[0] - (top_crop + bottom_crop)
#         new_width = img_array_o.shape[1] - (left_crop + right_crop)
#         new_array_o = img_array_o[top_crop:top_crop+new_height, left_crop:left_crop+new_width]

#         # new_array_o=cv2.resize(img_array_o,(img_length,img_width))
#         # blurred = cv2.GaussianBlur(img_array_o, (7, 7), 0)
#         (T, new_array_o) = cv2.threshold(new_array_o, 0, 255,cv2.THRESH_BINARY| cv2.THRESH_OTSU)

        
#         test_data.append([new_array_o])

# create_test_data(top_crop_o,bottom_crop_o,left_crop_o,right_crop_o)

# X is the inputs with the final patterns, y is the one with the images

X=training_data  
# y=test_data

X=(np.array(X).reshape(-1,1,img_shape_i,img_shape_i)) #/255.0  # last one is grayscale first minus one is all x
# y=(np.array(y).reshape(-1,1,img_shape_o,img_shape_o)) #/255.0

# normalizing images here to be bw 0 and 1 

X=X/255.0 
# y=y/255.0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device =  torch.device("cpu")


# Convert numpy arrays to torch tensors
X = torch.Tensor(X)
# y = torch.Tensor(y)


from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")


# torch_device = "cuda"
# vae.to(torch_device)

def encode_img(input_img):
    input_img = input_img.repeat(3, 1, 1)
    
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling  # to make outputs from -1 to 1 
    return 0.18215 * latent.latent_dist.sample()






def decode_img(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)    # to make outputs from 0 to 1 
    image = image.detach()
    return image


# Encode the images
encoded_latents = []




# encode latent of 100000 images

for img in range(0, X.shape[0]) :
    latent = encode_img(X[img,:,:,:])
    encoded_latents.append(latent)


# Simulating input image data as a list of random torch tensors
input_images_list=encoded_latents

# Convert the list of tensors to a numpy array
image_np = np.concatenate(input_images_list, axis=0)

# Now final_image_array contains the processed images with the desired properties
print("Shape of image array:",image_np.shape)

pickle_out=open("latent_dim_SimcorrtoExp_images_40900_4channels_4x32x32_07092024.pickle","wb")

pickle.dump(image_np,pickle_out)
pickle_out.close()