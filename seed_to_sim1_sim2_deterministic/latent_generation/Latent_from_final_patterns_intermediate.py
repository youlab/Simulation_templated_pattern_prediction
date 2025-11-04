# import libraries 

import os
import torch
import numpy as np
import pickle 

from models.vae import encode_img

from utils.config import SIMULATION_TEST_OUTPUT_FOLDER_2
from utils.config import LATENT_OUTPUT

from utils.preprocess import preprocess_simulation_output_data

'''
Loading training data, generating latents and saving as pickle file

'''

path_output = SIMULATION_TEST_OUTPUT_FOLDER_2  # for default patterns

# for building latents we use the entire dataset 
start_index = 0
end_index = -1

print(f"Loading data from: {path_output}")
print(f"start_index: {start_index}, end_index: {end_index}")

# Check how many files we have
all_files = sorted(os.listdir(path_output))
total_files = len(all_files)
print(f"Total files in directory: {total_files}")

# Process in larger chunks to avoid memory issues during loading
chunk_size = 1000  # Process 1000 images at a time during loading
batch_size = 96   # A5000 24GB trying for max usage without OOM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

encoded_latents = []
processed_count = 0

# Process data in manageable chunks
for chunk_start in range(0, total_files, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total_files)
    
    print(f"Loading chunk {chunk_start}-{chunk_end} ({chunk_end - chunk_start} images)")
    
    # Load chunk
    output_data = preprocess_simulation_output_data(path_output, chunk_start, chunk_end)
    
    if not output_data:
        print(f"No data in chunk {chunk_start}-{chunk_end}")
        continue
        
    # Convert chunk to tensor
    x_chunk = np.array(output_data).reshape(-1, 1, 256, 256) / 255.0
    x_chunk = torch.Tensor(x_chunk)
    
    print(f"Chunk shape: {x_chunk.shape}")
    
    # Process chunk in batches for VAE encoding
    for i in range(0, x_chunk.shape[0], batch_size):
        end_idx = min(i + batch_size, x_chunk.shape[0])
        batch = x_chunk[i:end_idx]
        
        processed_count += batch.shape[0]
        print(f"Processing batch {processed_count}/{total_files}")
        
        # Encode batch
        latent_batch = encode_img(batch)
        encoded_latents.append(latent_batch.cpu())
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Clear chunk from memory
    del output_data, x_chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"Processed {processed_count} images total")
# Convert the list of tensors to a numpy array
image_np = torch.cat(encoded_latents, dim=0).numpy()

# Now final_image_array contains the processed images with the desired properties
print("Shape of image array:",image_np.shape)


with open(LATENT_OUTPUT,'wb') as file:
    pickle.dump(image_np,file)

print(f"Saved latents to: {LATENT_OUTPUT}")


