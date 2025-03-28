import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.maze_dataset import MazeTensorDataset
from diffuser.encoder import VAE_Encoder
from diffuser.decoder import VAE_Decoder
from diffuser.model import Diffusion
from classifier_guidance.models.path_length_embedding import EmbeddingPathLengths
from vae_trainer import VAETrainer
from unet_trainer import UNetTrainer

CURR_FILE_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

# load json file
training_config = {}
with open(os.path.join(CURR_FILE_PARENT_DIR, "training_config.json"), "r") as f:
    traiing_config = json.load(f)

# vae training params
BATCH_SIZE = training_config.get("BATCH_SIZE", 128)
VAE_LEARNING_RATE = training_config.get("VAE_LEARNING_RATE", 1e-4)
VAE_EPOCHS = training_config.get("VAE_EPOCHS", 150)
VAE_LATENT_CHANNELS = training_config.get("VAE_LATENT_CHANNELS", 4)
VAE_MODELS_SAVE_PATH = training_config.get("VAE_MODELS_SAVE_PATH", "vae_model.pth")

# unet training params
UNET_EPOCHS = training_config.get("UNET_EPOCHS", 75)
NUM_TIMESTEPS = training_config.get("NUM_TIMESTEPS", 1000)
UNET_LATENT_CHANNELS = training_config.get("UNET_LATENT_CHANNELS", 4)
UNET_MODELS_SAVE_PATH = training_config.get("UNET_MODELS_SAVE_PATH", "unet_model.pth")

# classifier guidance params
GUIDANCE_SCALE = training_config.get("GUIDANCE_SCALE", 7.5)
PATH_EMBEDDING_CHOICE = training_config.get("PATH_EMBEDDING_CHOICE", "normal_path_length")  # or "inverse_path_length' or "exp_path_length" or "log_path_length"

EMBEDDING_CHOICE = {
    "normal_path_length": EmbeddingPathLengths().get_path_length_embedding,
    "inverse_path_length": EmbeddingPathLengths().get_inverse_path_length_embedding,
    "exp_path_length": EmbeddingPathLengths().get_exp_path_length_embedding,
    "log_path_length": EmbeddingPathLengths().get_log_path_length_embedding
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_vae(dataloader: DataLoader, test_dataset: MazeTensorDataset):
    # vae training and evaluation
    vae_trainer = VAETrainer()
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    encoder, decoder, optimizer, train_losses = vae_trainer.train(encoder, decoder, dataloader, device, VAE_EPOCHS, VAE_LATENT_CHANNELS, VAE_LEARNING_RATE)

    original_list, reconstructed_list, difference_list = vae_trainer.eval_model(encoder, decoder, test_dataset, device, VAE_LATENT_CHANNELS)
    vae_trainer.save_model(encoder, decoder, optimizer, os.path.join(CURR_FILE_PARENT_DIR, "data", "models", VAE_MODELS_SAVE_PATH))

def train_diffusion(dataloader: DataLoader, test_dataset: MazeTensorDataset, context_embedding_model: any):
    # diffusion training and evaluation
    unet_trainer = UNetTrainer(device, num_timesteps=NUM_TIMESTEPS)
    vae_encoder = VAE_Encoder().to(device).eval()
    decoder = VAE_Decoder().to(device).eval()
    checkpoint = torch.load(VAE_MODELS_SAVE_PATH, map_location=device)
    vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    diffusion_model = Diffusion().to(device)

    diffusion_model, train_losses = unet_trainer.train(diffusion_model, vae_encoder, UNET_EPOCHS, dataloader, UNET_LATENT_CHANNELS, GUIDANCE_SCALE, context_embedding_model)
    generated_image, test_img, test_path_length = unet_trainer.eval_model(diffusion_model, vae_encoder, test_dataset,context_embedding_model, None, 100)
    unet_trainer.save_model(diffusion_model, os.path.join(CURR_FILE_PARENT_DIR, "data", "models", UNET_MODELS_SAVE_PATH))

    # save train losses
    with open(os.path.join(CURR_FILE_PARENT_DIR, "data", "train_losses.json"), "w") as f:
        json.dump(train_losses, f)

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
    from generator.maze.maze_generator import generate_dataset
    from utils.data_preprocessor import load_dataset_from_npy, get_train_test_dataset

    # generate dataset if it doesn't exist
    data_path_mazes = os.path.join(CURR_FILE_PARENT_DIR, "data", "mazes")
    if not os.path.exists(data_path_mazes):
        generate_dataset(100_000, os.path.join(CURR_FILE_PARENT_DIR, "data"))

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

    # mean of train_path_lengths
    mean_train_path_lengths = sum(train_path_lengths) / len(train_path_lengths)
    variance_train_path_lengths = sum([(x - mean_train_path_lengths) ** 2 for x in train_path_lengths]) / len(train_path_lengths)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    if not os.path.exists(os.path.join(CURR_FILE_PARENT_DIR, "data", "models")):
        os.makedirs(os.path.join(CURR_FILE_PARENT_DIR, "data", "models"))
    train_vae(dataloader, test_dataset)
    context_embedding_model = EMBEDDING_CHOICE[PATH_EMBEDDING_CHOICE]
    train_diffusion(dataloader, test_dataset, context_embedding_model)