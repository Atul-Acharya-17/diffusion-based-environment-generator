import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from diffuser.encoder import VAE_Encoder
from diffuser.decoder import VAE_Decoder
from utils.maze_dataset import MazeTensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VAE_Encoder().to(device)
decoder = VAE_Decoder().to(device)

class VAETrainer:
    def __init__(self):
        pass

    def vae_loss(self, x, x_hat, mean, log_var):
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        
        kl_loss = 0.5 * (mean.pow(2) + log_var.exp() - log_var - 1).sum(dim=(1, 2, 3)).mean()
        
        return recon_loss + kl_loss

    def get_optimizer(self, lr: float) -> any:
        return optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    def train(self, encoder: VAE_Encoder, decoder: VAE_Decoder, dataloader: DataLoader, device: torch.device, epochs: int, latent_channels: int, lr: float) -> tuple[VAE_Encoder, VAE_Decoder, any, list[float]]:
        train_losses = []
        optimizer = self.get_optimizer(lr=lr)
        for epoch in range(epochs):
            encoder.train()
            decoder.train()
            train_loss = 0.0
            
            for _, x in enumerate(dataloader):
                
                x = x[0].to(device)
                batch_size = x.size(0)
                
                noise = torch.randn(batch_size, latent_channels, 8, 8).to(device)
                
                mean, log_var, z = encoder(x, noise)
                x_hat = decoder(z)
                
                loss = self.vae_loss(x, x_hat, mean, log_var)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_size
                
            train_loss = train_loss / len(dataloader.dataset)
            train_losses.append(train_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}')

        encoder.eval()
        decoder.eval()

        return encoder, decoder, optimizer, train_losses
    
    def eval_model(self, encoder: VAE_Encoder, decoder: VAE_Decoder, test_dataset: MazeTensorDataset, device: torch.device, latent_channels: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        original_list, reconstructed_list, difference_list = [], [], []
        with torch.no_grad():
            for i in range(5):
                idx = np.random.randint(len(test_dataset))
                img, _ = test_dataset[idx]  # Ignore the path length label for reconstruction
                img = img.unsqueeze(0).to(device)
                noise = torch.randn(1, latent_channels, 8, 8).to(device)
                
                _, _, z = encoder(img, noise)
                reconstructed = decoder(z).cpu().squeeze(0)
                
                original = img.cpu().squeeze(0)
                
                difference = torch.abs(original - reconstructed)

                original_list.append(original)
                reconstructed_list.append(reconstructed)
                difference_list.append(difference)
        
        return original_list, reconstructed_list, difference_list
    
    def save_model(self, encoder: VAE_Encoder, decoder: VAE_Decoder, optimizer: any, path: str):
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

        print("Model weights saved to path")