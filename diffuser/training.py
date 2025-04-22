import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
from maze_dataset.plotting import MazePlot
import datetime
import re
import importlib

# Change to project root directory and print current working directory
os.chdir("../")
print(f"Current working directory: {os.getcwd()}")

# --- Utility Functions ---
def get_timestamp():
    """Returns a timestamp string for unique file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Dataset and Preprocessing Functions ---
def preprocess_image(image, target_size=32):
    image = np.array(image)
    scale_factor = target_size // image.shape[0] 
    # image = np.kron(image, np.ones((scale_factor, scale_factor, 1))) 
    
    # image = image.astype(np.float32) / 127.5 - 1
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1)
    image = F.interpolate(image.unsqueeze(0), size=(target_size, target_size), mode='nearest').squeeze(0)  # (3, 32, 32)

    return image

def plot_grid_world(grid):
    """
    Plots the given grid world.
    """
    wall = grid[:,:,0] == 0
    source = grid[:,:,1] == 1
    destination = grid[:,:,2] == 1

    img = np.ones((*wall.shape, 3), dtype=np.float32)  # White background
    img[wall] = np.array([0, 0, 0])  # Walls → Black
    img[source] = np.array([1, 0, 0])  # Source → Red
    img[destination] = np.array([0, 1, 0])  # Destination → Green

    return img

def load_dataset_from_npy(parent_directory="./data", target_size=32, max_samples=20000):
    images = []
    path_lengths = []
    num_nodes_traversed_astar = []
    num_nodes_traversed_bfs = []
    
    mazes_directory = os.path.join(parent_directory, "mazes")
    files = sorted([f for f in os.listdir(mazes_directory) if f.endswith(".npy")])
    
    count = 0
    
    for file in files:
        img = np.load(os.path.join(mazes_directory, file))
        if(img.shape != (10,10,3)):
            continue
        # mask = np.all(img == [0, 0, 255], axis=-1)
        # img[mask] = [255, 255, 255]
        # img = img[:-1, :-1]
        # image = preprocess_image(img, target_size)

        image = plot_grid_world(img)
        mask = np.all(image == [0, 0, 255], axis=-1)
        image[mask] = [255, 255, 255]
        image = preprocess_image(image, target_size)

        pattern = r'maze_(\d+)'
        match = re.search(pattern, file)
        num = 0
        if match:
            num = int(match.group(1))
        else:
            continue
        
        # base_name = os.path.splitext(file)[0]
        # len_filename = base_name + "_len.txt"
        len_filename = f"path_length_{num}" + ".npy"
        len_path = os.path.join(mazes_directory, len_filename)
        astar_traversal_filename = f"a_star_{num}" + ".npy"
        astar_traversal_path = os.path.join(parent_directory, "a_star_l1_results" ,astar_traversal_filename)
        bfs_traversal_filename = f"bfs_{num}" + ".npy"
        bfs_traversal_path = os.path.join(parent_directory, "bfs_results" ,bfs_traversal_filename)
        
        # with open(len_path, "r") as f:
        #     maze_length = int(f.read().strip())
        maze_length = np.load(len_path)
        astar_traversal = np.load(astar_traversal_path)
        bfs_traversal = np.load(bfs_traversal_path)
        
        images.append(image)
        path_lengths.append(int(maze_length))
        num_nodes_traversed_astar.append(int(astar_traversal))
        num_nodes_traversed_bfs.append(int(bfs_traversal))

        count += 1
        if count >= max_samples:
            break  
    
    return images, path_lengths, num_nodes_traversed_astar, num_nodes_traversed_bfs

# , [a / b if b!=0 else a for a, b in zip(path_lengths, num_nodes_traversed_astar)], [a / b if b!=0 else a for a, b in zip(path_lengths, num_nodes_traversed_bfs)]

class MazeTensorDataset(Dataset):
    def __init__(self, images, path_lengths, num_nodes_astar, num_nodes_bfs):
        self.images = images
        self.path_lengths = path_lengths
        self.num_nodes_astar = num_nodes_astar
        self.num_nodes_bfs = num_nodes_bfs
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.path_lengths[idx], self.num_nodes_astar[idx], self.num_nodes_bfs[idx]

# --- Load Dataset ---
mazes_data_path = "./data"
images, org_path_lengths, num_nodes_astar, num_nodes_bfs = load_dataset_from_npy(mazes_data_path, target_size=32)

# # --- VAE Training Section ---
# Change to diffuser directory
os.chdir("./diffuser")
print(f"Current working directory: {os.getcwd()}")

# Reload encoder and decoder modules for VAE training
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from maze_dataset.plotting import MazePlot
import random

import importlib
import encoder
import decoder
importlib.reload(encoder)
importlib.reload(decoder)
from encoder import VAE_Encoder
from decoder import VAE_Decoder

# # VAE hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EPOCHS = 150
LATENT_CHANNELS = 8

path_lengths = org_path_lengths

# Split the dataset

total = len(images)
test_size = int(0.2 * total)
all_indices = list(range(total))
random.shuffle(all_indices)

test_indices = all_indices[:test_size]
train_indices = all_indices[test_size:]

train_images = [images[i] for i in train_indices]
train_path_lengths = [path_lengths[i] for i in train_indices]
train_num_nodes_astar = [num_nodes_astar[i] for i in train_indices]
train_num_nodes_bfs = [num_nodes_bfs[i] for i in train_indices]

test_images = [images[i] for i in test_indices]
test_path_lengths = [path_lengths[i] for i in test_indices]
test_num_nodes_astar = [num_nodes_astar[i] for i in test_indices]
test_num_nodes_bfs = [num_nodes_bfs[i] for i in test_indices]

dataset = MazeTensorDataset(train_images, train_path_lengths, train_num_nodes_astar, train_num_nodes_bfs)
test_dataset = MazeTensorDataset(test_images, test_path_lengths, test_num_nodes_astar, test_num_nodes_bfs)

print("Train dataset length:", len(dataset))
print("Test dataset length:", len(test_dataset))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

unique_train_paths = set(train_path_lengths)
print("Unique training path lengths:", unique_train_paths)
print("Number of unique training paths:", len(unique_train_paths))

unique_test_paths = set(test_path_lengths)
print("Unique test path lengths:", unique_test_paths)
print("Number of unique test paths:", len(unique_test_paths))

# Initialize VAE model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_encoder = VAE_Encoder().to(device)
vae_decoder = VAE_Decoder().to(device)
optimizer = optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=LEARNING_RATE)

def vae_loss(x, x_hat, mean, log_var):
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(dim=(1, 2, 3)).mean()
    return recon_loss + kl_loss

train_losses = []
val_losses = []
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Validation dataloader

for epoch in range(EPOCHS):
    vae_encoder.train()
    vae_decoder.train()
    train_loss = 0.0

    for x in dataloader:
        x = x[0].to(device)
        batch_size = x.size(0)
        # noise = torch.randn(batch_size, LATENT_CHANNELS, 8, 8).to(device)
        noise = torch.randn(batch_size, LATENT_CHANNELS, 16, 16).to(device)
        
        mean, log_var, z = vae_encoder(x, noise)
        x_hat = vae_decoder(z)
        loss = vae_loss(x, x_hat, mean, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    train_loss /= len(dataloader.dataset)
    train_losses.append(train_loss)

    # --- VAE Validation Pass ---
    vae_encoder.eval()
    vae_decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x in val_loader:
            x = x[0].to(device)
            # noise = torch.randn(x.size(0), LATENT_CHANNELS, 8, 8).to(device)
            noise = torch.randn(x.size(0), LATENT_CHANNELS, 16, 16).to(device)
            
            mean, log_var, z = vae_encoder(x, noise)
            x_hat = vae_decoder(z)
            loss = vae_loss(x, x_hat, mean, log_var)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot and save VAE training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Training Loss')
plt.legend()
vae_loss_curve_filename = f"loss_curve_VAE_{get_timestamp()}.png"
plt.savefig(vae_loss_curve_filename)

# Save VAE model weights with a timestamp
vae_weights_filename = f"vae_weights_{get_timestamp()}.pth"
torch.save({
    'encoder_state_dict': vae_encoder.state_dict(),
    'decoder_state_dict': vae_decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, vae_weights_filename)
print(f"Model weights saved to {vae_weights_filename}")

# --- Diffusion Training Section ---
# Diffusion hyperparameters
BATCH_SIZE = 256
EPOCHS = 50
NUM_TIMESTEPS = 1000
LATENT_CHANNELS = 8
LEARNING_RATE = 1e-5

# Split the dataset again
print("Total images:", len(images))
print("Total path_lengths:", len(path_lengths))

total = len(images)
test_size = int(0.2 * total)
all_indices = list(range(total))
random.shuffle(all_indices)

test_indices = all_indices[:test_size]
train_indices = all_indices[test_size:]

train_images = [images[i] for i in train_indices]
train_path_lengths = [path_lengths[i] for i in train_indices]
train_num_nodes_astar = [num_nodes_astar[i] for i in train_indices]
train_num_nodes_bfs = [num_nodes_bfs[i] for i in train_indices]

test_images = [images[i] for i in test_indices]
test_path_lengths = [path_lengths[i] for i in test_indices]
test_num_nodes_astar = [num_nodes_astar[i] for i in test_indices]
test_num_nodes_bfs = [num_nodes_bfs[i] for i in test_indices]

dataset = MazeTensorDataset(train_images, train_path_lengths, train_num_nodes_astar, train_num_nodes_bfs)
test_dataset = MazeTensorDataset(test_images, test_path_lengths, test_num_nodes_astar, test_num_nodes_bfs)

print("Train dataset length:", len(dataset))
print("Test dataset length:", len(test_dataset))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

unique_train_paths = set(train_path_lengths)
print("Unique training path lengths:", unique_train_paths)
print("Number of unique training paths:", len(unique_train_paths))

unique_test_paths = set(test_path_lengths)
print("Unique test path lengths:", unique_test_paths)
print("Number of unique test paths:", len(unique_test_paths))

# Reload required modules for Diffusion training
import importlib
import encoder
import decoder
import model
importlib.reload(encoder)
importlib.reload(decoder)
importlib.reload(model)
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from model import Diffusion
from ddpm import DDPMSampler
# from classifier_guidance.models import Classifier

# vae_weights_filename = "good_vae.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

vae_encoder = VAE_Encoder().to(device).eval()
vae_decoder = VAE_Decoder().to(device).eval()

# Load the pre-trained VAE weights saved earlier
checkpoint = torch.load(vae_weights_filename, map_location=device)
vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])


# Initialize Diffusion model, optimizer, and scheduler
diffusion_model = Diffusion(input_size=2).to(device)
optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = DDPMSampler(generator=torch.Generator(device=device), num_training_steps=NUM_TIMESTEPS)

val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_losses = []
val_losses = []
GUIDANCE_SCALE = 1.5
EARLY_STOPPING_PATIENCE = 5

best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    count = 0
    diffusion_model.train()

    for images, path_lengths, nodes_astar, nodes_bfs in tqdm(dataloader, desc=f"Training epoch [{epoch+1}/{EPOCHS}]"):
        images = images.to(device)
        path_lengths = path_lengths.clone().detach().float().to(device)
        nodes_astar = nodes_astar.clone().detach().float().to(device)
        zeros_path_tensor = torch.zeros_like(path_lengths).to(device)
        zeros_nodes_astar_tensor = torch.zeros_like(nodes_astar).to(device)

        # noise = torch.randn(images.size(0), LATENT_CHANNELS, 8, 8, device=device)
        noise = torch.randn(images.size(0), LATENT_CHANNELS, 16, 16, device=device)
        

        with torch.no_grad():
            _, _, z = vae_encoder(images, noise)

        timesteps = torch.randint(0, NUM_TIMESTEPS, (z.size(0),), device=device)
        noisy_z, noise_used = scheduler.add_noise(z, timesteps)

        noisy_z = noisy_z.repeat(2, 1, 1, 1)
        timesteps = timesteps.repeat_interleave(2)

        combined_features = torch.stack((path_lengths, nodes_astar), dim=-1)
        context = diffusion_model.condition_multidimensional_embedding(combined_features)

        combined_zeroes_features = torch.stack((zeros_path_tensor, zeros_nodes_astar_tensor), dim=-1)
        unconditional_guidance_embeddings = diffusion_model.condition_multidimensional_embedding(combined_zeroes_features)

        conditional_guidance_embeddings = torch.cat([unconditional_guidance_embeddings, context])
        noise_pred = diffusion_model(noisy_z, conditional_guidance_embeddings, timesteps)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        # loss = F.mse_loss(noise_pred, noise_used)
        loss = F.smooth_l1_loss(noise_pred, noise_used)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        count += images.size(0)

    avg_loss = epoch_loss / count
    train_losses.append(avg_loss)

    # --- Validation Pass ---
    diffusion_model.eval()
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, path_lengths, nodes_astar, nodes_bfs in val_loader:
            images = images.to(device)
            path_lengths = path_lengths.clone().detach().float().to(device)
            nodes_astar = nodes_astar.clone().detach().float().to(device)
            zeros_path_tensor = torch.zeros_like(path_lengths).to(device)
            zeros_nodes_astar_tensor = torch.zeros_like(nodes_astar).to(device)

            # noise = torch.randn(images.size(0), LATENT_CHANNELS, 8, 8, device=device)
            noise = torch.randn(images.size(0), LATENT_CHANNELS, 16, 16, device=device)
            
            _, _, z = vae_encoder(images, noise)
            timesteps = torch.randint(0, NUM_TIMESTEPS, (z.size(0),), device=device)
            noisy_z, noise_used = scheduler.add_noise(z, timesteps)

            noisy_z = noisy_z.repeat(2, 1, 1, 1)
            timesteps = timesteps.repeat_interleave(2)

            combined_features = torch.stack((path_lengths, nodes_astar), dim=-1)
            context = diffusion_model.condition_multidimensional_embedding(combined_features)

            combined_zeroes_features = torch.stack((zeros_path_tensor, zeros_nodes_astar_tensor), dim=-1)
            unconditional_guidance_embeddings = diffusion_model.condition_multidimensional_embedding(combined_zeroes_features)

            conditional_guidance_embeddings = torch.cat([unconditional_guidance_embeddings, context])
            noise_pred = diffusion_model(noisy_z, conditional_guidance_embeddings, timesteps)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

            # loss = F.mse_loss(noise_pred, noise_used)
            loss = F.smooth_l1_loss(noise_pred, noise_used)

            val_loss += loss.item() * images.size(0)
            count += images.size(0)

    val_loss /= count
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = {
            'epoch': epoch + 1,
            'diffusion_state_dict': diffusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        break

# Plot and save Diffusion training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, marker='x', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Training Loss')
plt.legend()
diffusion_loss_curve_filename = f"loss_curve_diffusion_{get_timestamp()}.png"
plt.savefig(diffusion_loss_curve_filename)

# Save Diffusion model weights with a timestamp
# --- Save Final or Best Model ---
if best_model_state:
    best_model_filename = f"best_diffusion_model_{get_timestamp()}.pth"
    torch.save(best_model_state, best_model_filename)
    print(f"Best model saved to {best_model_filename}")