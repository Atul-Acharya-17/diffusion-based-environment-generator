import random
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.data_preprocessor import load_dataset_from_npy, get_train_test_dataset
from utils.maze_dataset import MazeTensorDataset
from diffuser.encoder import VAE_Encoder
from diffuser.decoder import VAE_Decoder
from diffuser.model import Diffusion
from vae_trainer import VAETrainer
from unet_trainer import UNetTrainer

BATCH_SIZE = 128
VAE_LEARNING_RATE = 1e-4
VAE_EPOCHS = 150
VAE_LATENT_CHANNELS = 4
VAE_MODELS_SAVE_PATH = "./data/models/vae_model.pth"

UNET_EPOCHS = 75
NUM_TIMESTEPS = 1000
UNET_LATENT_CHANNELS = 4
UNET_MODELS_SAVE_PATH = "./data/models/unet_model.pth"

GUIDANCE_SCALE = 7.5

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # load images
    images, path_lengths = load_dataset_from_npy(directory="./data/mazes", target_size=32)
    dataset = MazeTensorDataset(images, path_lengths)
    print("Total images:", len(images))
    print("Total path_lengths:", len(path_lengths))

    # split dataset
    dataset, train_path_lengths, test_dataset, test_path_lengths = get_train_test_dataset(images, path_lengths)
    print("Train dataset length:", len(dataset))
    print("Test dataset length:", len(test_dataset))
    unique_train_paths = set(train_path_lengths)
    print("Unique training path lengths:", unique_train_paths)
    print("Number of unique training paths:", len(unique_train_paths))
    unique_test_paths = set(test_path_lengths)
    print("Unique test path lengths:", unique_test_paths)
    print("Number of unique test paths:", len(unique_test_paths))

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # vae training and evaluation
    vae_trainer = VAETrainer()
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    encoder, decoder, optimizer, train_losses = vae_trainer.train(encoder, decoder, dataloader, device, VAE_EPOCHS, VAE_LATENT_CHANNELS, VAE_LEARNING_RATE)

    original_list, reconstructed_list, difference_list = vae_trainer.eval_model(encoder, decoder, test_dataset, device, VAE_LATENT_CHANNELS)

    vae_trainer.save_model(encoder, decoder, optimizer, VAE_MODELS_SAVE_PATH)

    # diffusion training and evaluation
    unet_trainer = UNetTrainer(device, num_timesteps=NUM_TIMESTEPS)
    vae_encoder = VAE_Encoder().to(device).eval()
    decoder = VAE_Decoder().to(device).eval()
    checkpoint = torch.load(VAE_MODELS_SAVE_PATH, map_location=device)
    vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    diffusion_model = Diffusion().to(device)

    diffusion_model, train_losses = unet_trainer.train(diffusion_model, vae_encoder, UNET_EPOCHS, dataloader, UNET_LATENT_CHANNELS, GUIDANCE_SCALE)
    generated_image, test_img, test_path_length = unet_trainer.eval_model(diffusion_model, vae_encoder, test_dataset, UNET_LATENT_CHANNELS, GUIDANCE_SCALE)
    unet_trainer.save_model(diffusion_model, UNET_MODELS_SAVE_PATH)

    # plot original and generated mazes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(generated_image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title(f"Generated Maze, {test_path_length}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(test_img.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.title("Original Test Maze")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()