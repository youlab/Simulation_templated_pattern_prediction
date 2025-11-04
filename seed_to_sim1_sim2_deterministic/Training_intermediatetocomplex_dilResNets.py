import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pickle
from datetime import datetime
import time
import json
import os

from models.dilResNet import PDEArenaDilatedResNet

from utils.config import MODEL_SAVE_LOCATION,LOGS_SAVE_LOCATION
from utils.config import LATENT_OUTPUT_SAVED, LATENT_OUTPUT_TDB_SAVED

########### SLURM Stuff ##############
taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])
#######################################


currentDay = datetime.now().day
currentMonth = datetime.now().month

NO_OF_SAMPLES=30000 # number of samples to load for training
BATCH_SIZE=64 # after looking at nvtop
NAME =f"Pixel_32x32x3to32x32x4_dilRESNET_graypatterns_intermediatetocomplex_Model_{NO_OF_SAMPLES}_v{currentMonth}{currentDay}_{taskID}-{int(time.time())}"  # change this later to incorporate exact date 


# Load saved latent representation, act as input and output for training

pickle_in=open(LATENT_OUTPUT_SAVED,"rb")
yprime_in=pickle.load(pickle_in)
yprime_in=yprime_in[:NO_OF_SAMPLES,:,:,:]


pickle_in_output=open(LATENT_OUTPUT_TDB_SAVED,"rb")
yprime_in_output=pickle.load(pickle_in_output)
yprime_in_output=yprime_in_output[:NO_OF_SAMPLES,:,:,:]


print(f"Loaded input samples shape: {yprime_in.shape}")
print(f"Loaded output samples shape: {yprime_in_output.shape}")

yprime_in=torch.Tensor(yprime_in)
yprime_in_output=torch.Tensor(yprime_in_output)

yprime_scaled_in=yprime_in
yprime_scaled_in_output=yprime_in_output

# yprime_scaled=yprime
yprime_scaled_in=yprime_scaled_in.float()
yprime_scaled_in_output=yprime_scaled_in_output.float()


# Define train and test datasets
dataset = torch.utils.data.TensorDataset(yprime_scaled_in, yprime_scaled_in_output)
# Split dataset into train and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)


# Example usage
model = PDEArenaDilatedResNet(
    in_channels=4,               # Input channels (e.g., RGB image)
    out_channels=4,              # Output channels (e.g., RGB image or latent channels)
    hidden_channels=64,          # Number of hidden channels
    num_blocks=18,               # Number of dilated blocks (similar to number of ResNet blocks)
    dilation_rates=[1, 2, 4, 8], # Dilation rates for multi-scale feature capture
    activation=nn.ReLU,          # Activation function
    norm=True                    # Use BatchNorm after each convolution
)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_epochs = 500       
warmup_epochs=10
lr = 5e-4               
min_lr = 5e-6
gamma = 0.99

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
print(f"Total Parameters in Neural Network: {count_parameters(model)}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)


print("Starting training...")

# Training parameters and early stopping initialization # to save best epoch
best_loss = float('inf')
best_epoch=0
epochs_without_improvement = 0
patience = 70
train_losses=[]
test_losses=[]
saved_model_epoch=0
delta=0.05


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_params, batch_latents in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_params.to(device))
        loss = criterion(outputs, batch_latents.squeeze(1).to(device))

        loss.backward()

        # Warm-up schedule
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup_epochs

        optimizer.step()
        running_loss += loss.item()

    # Scheduler step after warmup
    if epoch >= warmup_epochs:
        scheduler.step()
    param_group['lr'] = max(param_group['lr'], min_lr)

    # Validation loop for testing set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_params, batch_latents in test_loader:
            outputs = model(batch_params.to(device))
            loss = criterion(outputs, batch_latents.squeeze(1).to(device))
            test_loss += loss.item()

    # Compute average losses for this epoch
    avg_train_loss = running_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    avg_val_loss=avg_test_loss

    # Store losses in lists
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    interval = 2 if epoch < 20 else 40
    if (epoch + 1) % interval == 0 or epoch + 1 == num_epochs:
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | lr: {param_group['lr']:0.7f}")

    # Early stopping logic with tolerance
    if avg_val_loss < best_loss :  # remove delta for strict improvement
        # Significant improvement
        best_loss = avg_val_loss
        epochs_without_improvement = 0
        best_epoch = epoch
        print(f"Epoch {epoch + 1}: Significant improvement observed. Best Validation Loss updated to {best_loss:.6f}.")
        # save the best model weights
        torch.save(model.state_dict(), f'{MODEL_SAVE_LOCATION}/{NAME}_best.pt')

    elif avg_val_loss <= best_loss + delta:
        # Within tolerance
        epochs_without_improvement = 0
        print(f"Epoch {epoch + 1}: Validation loss increased but within tolerance ({delta}). Continuing training.")
    else:
        # Exceeded tolerance
        epochs_without_improvement += 1
        print(f"Epoch {epoch + 1}: Validation loss increased beyond tolerance. Epochs without improvement: {epochs_without_improvement}/{patience}.")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement after {patience} epochs.")
            break



torch.save(model.state_dict(), f'{MODEL_SAVE_LOCATION}/{NAME}.pt') 

# plot losses

# Save losses and model details in a JSON file at the end of training
losses = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_loss': best_loss
    # 'saved_model_epoch': saved_model_epoch,
    # 'model_name': NAME
}

with open(f'{LOGS_SAVE_LOCATION}/losses_{NAME}.json', 'w') as f:
    json.dump(losses, f, indent=4)



