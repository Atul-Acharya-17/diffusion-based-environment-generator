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
    image = np.array(image, dtype=np.uint8)  # Ensure it's an ndarray
    color_map = {
        (0, 0, 0): (0, 0, 0),         # Wall → Black
        (1, 0, 0): (255, 255, 255),   # Empty → White
        (1, 1, 0): (0, 128, 0),       # Source → Green
        (1, 0, 1): (255, 0, 0),       # End → Red 
    }
    h, w, c = image.shape
    processed_image = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            pixel = tuple(image[i, j])
            processed_image[i, j] = color_map.get(pixel, (128, 128, 128))
    resized_image = cv2.resize(processed_image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    resized_image = resized_image.astype(np.float32) / 127.5 - 1.0 
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1)
    return resized_image

def load_dataset_from_npy(directory="./data", target_size=32):
    images = []
    path_lengths = []
    maze_files = sorted([f for f in os.listdir(directory) if f.startswith("maze_") and f.endswith(".npy")])
    for maze_file in tqdm(maze_files, desc="Loading Mazes", unit="maze"):
        img = np.load(os.path.join(directory, maze_file))
        image = preprocess_image(img, target_size)
        path_length_file = maze_file.replace("maze_", "path_length_")
        path_length_path = os.path.join(directory, path_length_file)
        if os.path.exists(path_length_path):
            path_length = np.load(path_length_path).item()
        else:
            path_length = -1  
        images.append(image)
        path_lengths.append(path_length)
    return images, path_lengths

class MazeTensorDataset(Dataset):
    def __init__(self, images, path_lengths):
        self.images = images
        self.path_lengths = path_lengths
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.path_lengths[idx]

# --- Load Dataset and Visualize ---
mazes_data_path = "./data"
images, path_lengths = load_dataset_from_npy(mazes_data_path, target_size=32)
# Display the first image for verification
img = images[0].permute(1, 2, 0).cpu().numpy()
img = ((img + 1.0) * 127.5).astype(np.uint8)
print(img.shape)
print(path_lengths[0])
plt.imshow(img)
plt.axis('off')
plt.show()

def split_dataset(images, path_lengths, test_ratio=0.2):
    total = len(images)
    test_size = int(test_ratio * total)
    all_indices = list(range(total))
    random.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    train_indices = all_indices[test_size:]
    train_images = [images[i] for i in train_indices]
    train_path_lengths = [path_lengths[i] for i in train_indices]
    test_images = [images[i] for i in test_indices]
    test_path_lengths = [path_lengths[i] for i in test_indices]
    return train_images, train_path_lengths, test_images, test_path_lengths

# # --- VAE Training Section ---
# # Change to diffuser directory
# os.chdir("./diffuser")
# print(f"Current working directory: {os.getcwd()}")

# # Reload encoder and decoder modules for VAE training
# import encoder
# import decoder
# importlib.reload(encoder)
# importlib.reload(decoder)
# from encoder import VAE_Encoder
# from decoder import VAE_Decoder

# # VAE hyperparameters
# BATCH_SIZE = 256
# LEARNING_RATE = 1e-4
# EPOCHS = 150
# LATENT_CHANNELS = 4

# print("Total images:", len(images))
# print("Total path_lengths:", len(path_lengths))

# # Split the dataset
# train_images, train_path_lengths, test_images, test_path_lengths = split_dataset(images, path_lengths, test_ratio=0.2)
# dataset = MazeTensorDataset(train_images, train_path_lengths)
# test_dataset = MazeTensorDataset(test_images, test_path_lengths)
# print("Train dataset length:", len(dataset))
# print("Test dataset length:", len(test_dataset))
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# print("Unique training path lengths:", set(train_path_lengths), "Count:", len(set(train_path_lengths)))
# print("Unique test path lengths:", set(test_path_lengths), "Count:", len(set(test_path_lengths)))

# # Initialize VAE model and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae_encoder = VAE_Encoder().to(device)
# vae_decoder = VAE_Decoder().to(device)
# optimizer = optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=LEARNING_RATE)

# def vae_loss(x, x_hat, mean, log_var):
#     recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
#     kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(dim=(1, 2, 3)).mean()
#     return recon_loss + kl_loss

# train_losses = []
# for epoch in range(EPOCHS):
#     vae_encoder.train()
#     vae_decoder.train()
#     train_loss = 0.0
#     for batch_idx, (x, _) in enumerate(dataloader):
#         x = x.to(device)
#         batch_size = x.size(0)
#         noise = torch.randn(batch_size, LATENT_CHANNELS, 8, 8, device=device)
#         mean, log_var, z = vae_encoder(x, noise)
#         x_hat = vae_decoder(z)
#         loss = vae_loss(x, x_hat, mean, log_var)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * batch_size
#     train_loss /= len(dataloader.dataset)
#     train_losses.append(train_loss)
#     print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}')

# # Plot and save VAE training loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('VAE Training Loss')
# plt.legend()
# vae_loss_curve_filename = f"loss_curve_VAE_{get_timestamp()}.png"
# plt.savefig(vae_loss_curve_filename)

# # Save VAE model weights with a timestamp
# vae_weights_filename = f"vae_weights_{get_timestamp()}.pth"
# torch.save({
#     'encoder_state_dict': vae_encoder.state_dict(),
#     'decoder_state_dict': vae_decoder.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, vae_weights_filename)
# print(f"Model weights saved to {vae_weights_filename}")

# --- Diffusion Training Section ---
# Diffusion hyperparameters
BATCH_SIZE = 128
EPOCHS = 150
NUM_TIMESTEPS = 1000
LATENT_CHANNELS = 4

print("Total images:", len(images))
print("Total path_lengths:", len(path_lengths))

# Split the dataset again
train_images, train_path_lengths, test_images, test_path_lengths = split_dataset(images, path_lengths, test_ratio=0.2)
dataset = MazeTensorDataset(train_images, train_path_lengths)
test_dataset = MazeTensorDataset(test_images, test_path_lengths)
print("Train dataset length:", len(dataset))
print("Test dataset length:", len(test_dataset))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Unique training path lengths:", set(train_path_lengths), "Count:", len(set(train_path_lengths)))
print("Unique test path lengths:", set(test_path_lengths), "Count:", len(set(test_path_lengths)))

# Reload required modules for Diffusion training
import model
import ddpm
importlib.reload(encoder)
importlib.reload(decoder)
importlib.reload(model)
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from model import Diffusion
from ddpm import DDPMSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae_weights_filename = "vae_weights_20250326_065408.pth"

# Load the pre-trained VAE weights saved earlier
vae_encoder = VAE_Encoder().to(device).eval()
vae_decoder = VAE_Decoder().to(device).eval()
checkpoint = torch.load(vae_weights_filename, map_location=device)
vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])

# Initialize Diffusion model, optimizer, and scheduler
diffusion_model = Diffusion().to(device)
optimizer_diffusion = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = DDPMSampler(generator=torch.Generator(device=device), num_training_steps=NUM_TIMESTEPS)

train_losses_diffusion = []
GUIDANCE_SCALE = 10

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    count = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
    for batch_idx, (imgs, path_lengths_batch) in progress_bar:
        imgs = imgs.to(device)
        path_lengths_batch = path_lengths_batch.float().to(device)
        zeros_tensor = torch.zeros_like(path_lengths_batch).to(device)
        noise = torch.randn(imgs.size(0), LATENT_CHANNELS, 8, 8, device=device)
        with torch.no_grad():
            _, _, z = vae_encoder(imgs, noise)
        timesteps = torch.randint(0, NUM_TIMESTEPS, (z.size(0),), device=device)
        noisy_z, noise_used = scheduler.add_noise(z, timesteps)
        noisy_z = noisy_z.repeat(2, 1, 1, 1)
        context = diffusion_model.condition_embedding(path_lengths_batch)
        unconditional_guidance_embeddings = diffusion_model.condition_embedding(zeros_tensor)
        conditional_guidance_embeddings = torch.cat([unconditional_guidance_embeddings, context])
        cond_timesteps = torch.cat([timesteps, timesteps], dim=0)
        noise_pred = diffusion_model(noisy_z, conditional_guidance_embeddings, cond_timesteps)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
        # loss = F.mse_loss(noise_pred, noise_used)
        
        optimizer_diffusion.zero_grad()
        loss.backward()
        optimizer_diffusion.step()
        epoch_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = epoch_loss / count if count > 0 else 0.0
    train_losses_diffusion.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")

# Plot and save Diffusion training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses_diffusion, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Graph')
plt.legend()
diffusion_loss_curve_filename = f"loss_curve_diffusion_{get_timestamp()}.png"
plt.savefig(diffusion_loss_curve_filename)

# Save Diffusion model weights with a timestamp
diffusion_weights_filename = f"diffusion_weights_{get_timestamp()}.pth"
torch.save({
    'diffusion_state_dict': diffusion_model.state_dict(),
    'optimizer_state_dict': optimizer_diffusion.state_dict(),
    'train_losses': train_losses_diffusion
}, diffusion_weights_filename)
print(f"Diffusion weights saved to {diffusion_weights_filename}")

# import os
# import random
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import cv2

# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset
# from torch.nn import functional as F
# from PIL import Image
# from maze_dataset.plotting import MazePlot

# os.chdir("../")
# print(f"Current working directory: {os.getcwd()}")

# def preprocess_image(image, target_size=32):
#     image = np.array(image, dtype=np.uint8)  # Ensure it's an ndarray

#     color_map = {
#         (0, 0, 0): (0, 0, 0),         # Wall → Black
#         (1, 0, 0): (255, 255, 255),   # Empty → White
#         (1, 1, 0): (0, 128, 0),       # Source → Green
#         (1, 0, 1): (255, 0, 0),       # End → Red 
#     }

#     h, w, c = image.shape
#     processed_image = np.zeros((h, w, c), dtype=np.uint8)

#     for i in range(h):
#         for j in range(w):
#             pixel = tuple(image[i, j])
#             processed_image[i, j] = color_map.get(pixel, (128, 128, 128))  
    
#     resized_image = cv2.resize(processed_image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
#     # resized_image = processed_image
#     resized_image = resized_image.astype(np.float32) / 127.5 - 1.0 
    
#     resized_image = torch.from_numpy(resized_image).permute(2, 0, 1) 

#     return resized_image

# def load_dataset_from_npy(directory="./data", target_size=32):
#     images = []
#     path_lengths = []
    
#     maze_files = sorted([f for f in os.listdir(directory) if f.startswith("maze_") and f.endswith(".npy")])
    
#     for maze_file in tqdm(maze_files, desc="Loading Mazes", unit="maze"):
#         img = np.load(os.path.join(directory, maze_file))
#         image = preprocess_image(img, target_size)
        
#         path_length_file = maze_file.replace("maze_", "path_length_")
#         path_length_path = os.path.join(directory, path_length_file)
        
#         if os.path.exists(path_length_path):
#             path_length = np.load(path_length_path).item()
#         else:
#             path_length = -1  
        
#         images.append(image)
#         path_lengths.append(path_length)
#     return images, path_lengths



# # mazes_data_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data", "mazes")
# mazes_data_path = "./data"
# images, path_lengths = load_dataset_from_npy(mazes_data_path, target_size=32)

# img = images[0].permute(1, 2, 0).cpu().numpy()  
# img = ((img + 1.0) * 127.5).astype(np.uint8)
# print(img.shape)
# print(path_lengths[0])

# plt.imshow(img)
# plt.axis('off')
# plt.show()

# class MazeTensorDataset(Dataset):
#     def __init__(self, images, path_lengths):
#         self.images = images
#         self.path_lengths = path_lengths
        
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         return self.images[idx], self.path_lengths[idx]

# os.chdir("./diffuser")
# print(f"Current working directory: {os.getcwd()}")

# # VAE TRAINING
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from maze_dataset.plotting import MazePlot
# import random
# import importlib
# import encoder
# import decoder
# importlib.reload(encoder)
# importlib.reload(decoder)
# from encoder import VAE_Encoder
# from decoder import VAE_Decoder

# # Hyperparameters FOR VAE
# BATCH_SIZE = 256
# LEARNING_RATE = 1e-4
# EPOCHS = 150
# LATENT_CHANNELS = 4

# print("Total images:", len(images))
# print("Total path_lengths:", len(path_lengths))

# total = len(images)
# test_size = int(0.2 * total)
# all_indices = list(range(total))
# random.shuffle(all_indices)

# test_indices = all_indices[:test_size]
# train_indices = all_indices[test_size:]

# train_images = [images[i] for i in train_indices]
# train_path_lengths = [path_lengths[i] for i in train_indices]

# test_images = [images[i] for i in test_indices]
# test_path_lengths = [path_lengths[i] for i in test_indices]

# dataset = MazeTensorDataset(train_images, train_path_lengths)
# test_dataset = MazeTensorDataset(test_images, test_path_lengths)

# print("Train dataset length:", len(dataset))
# print("Test dataset length:", len(test_dataset))

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# unique_train_paths = set(train_path_lengths)
# print("Unique training path lengths:", unique_train_paths)
# print("Number of unique training paths:", len(unique_train_paths))

# unique_test_paths = set(test_path_lengths)
# print("Unique test path lengths:", unique_test_paths)
# print("Number of unique test paths:", len(unique_test_paths))

# # Model Initialization
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# encoder = VAE_Encoder().to(device)
# decoder = VAE_Decoder().to(device)
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# def vae_loss(x, x_hat, mean, log_var):
#     recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    
#     kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(dim=(1, 2, 3)).mean()
    
#     return recon_loss + kl_loss

# train_losses = []
# for epoch in range(EPOCHS):
#     encoder.train()
#     decoder.train()
#     train_loss = 0.0
    
#     for batch_idx, x in enumerate(dataloader):
        
#         x = x[0].to(device)
#         batch_size = x.size(0)
        
#         noise = torch.randn(batch_size, LATENT_CHANNELS, 8, 8).to(device)
        
#         mean, log_var, z = encoder(x, noise)
#         x_hat = decoder(z)
        
#         loss = vae_loss(x, x_hat, mean, log_var)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item() * batch_size
        
#     train_loss = train_loss / len(dataloader.dataset)
#     train_losses.append(train_loss)
#     print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}')

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('VAE Training Loss')
# plt.legend()
# plt.savefig("loss_curve_VAE.png")

# # Save Model Weights
# torch.save({
#     'encoder_state_dict': encoder.state_dict(),
#     'decoder_state_dict': decoder.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, 'vae_weights_1.pth')

# print("Model weights saved to vae_weights_1.pth")


# # Hyperparameters
# BATCH_SIZE = 256
# EPOCHS = 75
# NUM_TIMESTEPS = 1000
# LATENT_CHANNELS = 4


# print("Total images:", len(images))
# print("Total path_lengths:", len(path_lengths))

# total = len(images)
# test_size = int(0.2 * total)
# all_indices = list(range(total))
# random.shuffle(all_indices)

# test_indices = all_indices[:test_size]
# train_indices = all_indices[test_size:]

# train_images = [images[i] for i in train_indices]
# train_path_lengths = [path_lengths[i] for i in train_indices]

# test_images = [images[i] for i in test_indices]
# test_path_lengths = [path_lengths[i] for i in test_indices]

# dataset = MazeTensorDataset(train_images, train_path_lengths)
# test_dataset = MazeTensorDataset(test_images, test_path_lengths)

# print("Train dataset length:", len(dataset))
# print("Test dataset length:", len(test_dataset))

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# unique_train_paths = set(train_path_lengths)
# print("Unique training path lengths:", unique_train_paths)
# print("Number of unique training paths:", len(unique_train_paths))

# unique_test_paths = set(test_path_lengths)
# print("Unique test path lengths:", unique_test_paths)
# print("Number of unique test paths:", len(unique_test_paths))


# import importlib
# import encoder
# import decoder
# import model
# importlib.reload(encoder)
# importlib.reload(decoder)
# importlib.reload(model)
# from encoder import VAE_Encoder
# from decoder import VAE_Decoder
# from model import Diffusion
# from ddpm import DDPMSampler
# # from classifier_guidance.models import Classifier

# device = "cuda" if torch.cuda.is_available() else "cpu"

# vae_encoder = VAE_Encoder().to(device).eval()
# decoder = VAE_Decoder().to(device).eval()

# checkpoint = torch.load('vae_weights_1.pth', map_location=device)
# vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
# decoder.load_state_dict(checkpoint['decoder_state_dict'])

# diffusion_model = Diffusion().to(device)
# optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4, weight_decay=0.01)
# scheduler = DDPMSampler(generator=torch.Generator(device=device), num_training_steps=NUM_TIMESTEPS)


# train_losses = []
# GUIDANCE_SCALE = 7.5

# for epoch in range(EPOCHS):
#     epoch_loss = 0.0
#     count = 0

#     progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

#     for batch_idx, (images, path_lengths) in progress_bar:
#         images = images.to(device)
#         path_lengths = path_lengths.float().to(device)
#         zeros_tensor = torch.zeros_like(path_lengths).to(device)
        
#         noise = torch.randn(images.size(0), LATENT_CHANNELS, 8, 8, device=device)
        
#         with torch.no_grad():
#             _, _, z = vae_encoder(images, noise)
        
#         timesteps = torch.randint(0, NUM_TIMESTEPS, (z.size(0),), device=device)
#         noisy_z, noise_used = scheduler.add_noise(z, timesteps)

#         noisy_z = noisy_z.repeat(2, 1, 1, 1)

#         context = diffusion_model.condition_embedding(path_lengths)
#         unconditional_guidance_embeddings = diffusion_model.condition_embedding(zeros_tensor)
#         conditional_guidance_embeddings = torch.cat([unconditional_guidance_embeddings, context])
        
#         cond_timesteps = torch.cat([timesteps, timesteps], dim=0)
#         noise_pred = diffusion_model(noisy_z, conditional_guidance_embeddings, cond_timesteps)
#         noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
        
#         loss = F.mse_loss(noise_pred, noise_used)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item() * images.size(0)
#         count += images.size(0)

#         # Update tqdm with current loss
#         progress_bar.set_postfix(loss=loss.item())

#     avg_loss = epoch_loss / count if count > 0 else 0.0
#     train_losses.append(avg_loss)
#     print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Graph')
# plt.legend()
# plt.savefig("loss_curve_150.png")

# torch.save({
#     'diffusion_state_dict': diffusion_model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'train_losses': train_losses
# }, 'diffusion_weights_150.pth')
# print("Diffusion weights saved to diffusion_weights_150.pth")