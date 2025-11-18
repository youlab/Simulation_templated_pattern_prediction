import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json 
from datetime import datetime
import time


from models.dilResNet import PDEArenaDilatedResNet

from utils.config import MODEL_SAVE_LOCATION,LOGS_SAVE_LOCATION
from utils.config import LATENT_OUTPUT_BASE

# taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])
# size_array=np.array([10,20,40,60,80,100,200,400,800,1600])
# end_pattern= size_array[taskID-1]

# modification since 800 got a device that did not have a gpu

end_pattern=np.array(800)

currentDay = datetime.now().day
currentMonth = datetime.now().month

LATENT_INTERMEDIATE_FILE= f'{LATENT_OUTPUT_BASE}/latent_dim_AugmentedDataDemand_4channels_4x32x32_{end_pattern}_intermediate_926.pickle'
LATENT_COMPLEX_FILE= f'{LATENT_OUTPUT_BASE}/latent_dim_AugmentedDataDemand_4channels_4x32x32_{end_pattern}_complex_926.pickle'

NAME =f"dilRESNET_DataAugmentation_intermediatetocomplex_{end_pattern}_v{currentMonth}{currentDay}-{int(time.time())}"  # change this later to incorporate exact date 
BATCH_SIZE = 64

#MODIFIED PICKLES ON 093025 TO NEW GRAY PATTERN LATENTS
pickle_in=open(LATENT_INTERMEDIATE_FILE,"rb")
yprime_in=pickle.load(pickle_in)


pickle_in_output=open(LATENT_COMPLEX_FILE,"rb")
yprime_in_output=pickle.load(pickle_in_output)

yprime_in=torch.Tensor(yprime_in)
yprime_in_output=torch.Tensor(yprime_in_output)


yprime_scaled_in=yprime_in
yprime_scaled_in_output=yprime_in_output


yprime_scaled_in=yprime_scaled_in.float()
yprime_scaled_in_output=yprime_scaled_in_output.float()


print(f"Loaded input samples shape: {yprime_scaled_in.shape}")
print(f"Loaded output samples shape: {yprime_scaled_in_output.shape}")

# Define train and test datasets
dataset = torch.utils.data.TensorDataset(yprime_scaled_in, yprime_scaled_in_output)

# Split dataset into train and validation sets (simple index-based split)
train_size = int(0.9 * len(dataset))

# Direct tensor slicing for simple index-based split
# note this is done to avoid random_split shuffling which can lead to data leakage due to augmentation
train_inputs = yprime_scaled_in[:train_size]
train_outputs = yprime_scaled_in_output[:train_size]
val_inputs = yprime_scaled_in[train_size:]
val_outputs = yprime_scaled_in_output[train_size:]

# Create train and validation datasets
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
val_dataset = torch.utils.data.TensorDataset(val_inputs, val_outputs)


train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

# Example usage
model = PDEArenaDilatedResNet(
    in_channels=4,               # Input channels 
    out_channels=4,              # Output channels 
    hidden_channels=64,          # Number of hidden channels
    num_blocks=18,               # Number of dilated blocks 
    dilation_rates=[1, 2, 4, 8], # Dilation rates for multi-scale feature capture
    activation=nn.ReLU,          # Activation function
    norm=True                    # Use BatchNorm after each convolution
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_epochs = 500      
warmup_epochs=10
lr = 5e-4               #for fine tuning.
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

# Training parameters and early stopping initialization

print("Starting training...")

# Training parameters and early stopping initialization # to save best epoch
best_loss = float('inf')
tolerant_loss= None
best_epoch=0
epochs_without_improvement = 0
patience = 70
train_losses=[]
test_losses=[]
saved_model_epoch=0
delta=0.05
train_delta=0.01
best_train_for_tolerant = float('inf')  # NEW
tolerant_epoch=None


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
        best_train_for_tolerant = avg_train_loss  # reset anchor
        torch.save(model.state_dict(), f'{MODEL_SAVE_LOCATION}/{NAME}_best.pt')
        print(f"Epoch {epoch + 1}: New BEST (val): {best_loss:.6f}")

    elif (avg_val_loss <= best_loss + delta) and (avg_train_loss < best_train_for_tolerant - train_delta):
        # Within tolerance
        epochs_without_improvement = 0
        best_train_for_tolerant = avg_train_loss
        tolerant_loss=avg_val_loss
        torch.save(model.state_dict(), f'{MODEL_SAVE_LOCATION}/{NAME}_best_tolerant.pt')
        tolerant_epoch=epoch +1
        print(f"Epoch {epoch + 1}: Saved tolerant best (val {avg_val_loss:.6f} within {delta}, "
                f"train improved by ≥{train_delta}).")
    else:
        # Exceeded tolerance
        epochs_without_improvement += 1
        print(f"Epoch {epoch + 1}: Validation loss increased beyond tolerance. Epochs without improvement: {epochs_without_improvement}/{patience}.")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement after {patience} epochs.")
            break



torch.save(model.state_dict(), f'{MODEL_SAVE_LOCATION}/{NAME}.pt')

# Save losses and model details in a JSON file at the end of training
losses = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_loss': best_loss,
    'tolerant_loss': tolerant_loss,
    'best_epoch': best_epoch + 1,
    'tolerant_epoch': tolerant_epoch
}

with open(f'{LOGS_SAVE_LOCATION}/losses_{NAME}.json', 'w') as f:
    json.dump(losses, f, indent=4)

print(f"Training complete. Best Validation Loss: {best_loss:.6f} at epoch {best_epoch + 1}. Model saved as {MODEL_SAVE_LOCATION}/{NAME}.pt")
if tolerant_epoch is not None:
    print(f"Tolerant best model at epoch {tolerant_epoch} saved as {MODEL_SAVE_LOCATION}/{NAME}_best_tolerant.pt")
else:
    print("No tolerant-best model was saved.")