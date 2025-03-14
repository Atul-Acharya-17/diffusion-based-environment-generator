import torch
from torch.utils.data import DataLoader

from diffuser.encoder import VAE_Encoder
from diffuser.decoder import VAE_Decoder
from diffuser.model import Diffusion
from diffuser.ddpm import DDPMSampler
from utils.maze_dataset import MazeTensorDataset

class UNetTrainer():
    def __init__(self, device: torch.device, num_timesteps: int):
        self.device = device
        self.num_timesteps = num_timesteps

        # self.vae_encoder = VAE_Encoder().to(self.device).eval()
        # self.decoder = VAE_Decoder().to(self.device).eval()
        # checkpoint = torch.load(vae_model_path, map_location=self.device)
        # self.vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        # self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # self.diffusion_model = Diffusion().to(device)
        self.scheduler = DDPMSampler(generator=torch.Generator(device=device), num_training_steps=self.num_timesteps)

    def train(self, diffusion_model:Diffusion, vae_encoder: VAE_Encoder, epochs: int, dataloader: DataLoader, latent_channels: int, guidance_scale: float):
        train_losses = []
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-4, weight_decay=0.01)

        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0
            for _, (images, path_lengths) in enumerate(dataloader):
                images = images.to(self.device)
                path_lengths = torch.tensor(path_lengths).float().to(self.device)
                zeros_tensor = torch.zeros_like(path_lengths).to(self.device)
                
                noise = torch.randn(images.size(0), latent_channels, 8, 8, device=self.device)
                
                with torch.no_grad():
                    _, _, z = vae_encoder(images, noise)
                
                timesteps = torch.randint(0, self.num_timesteps, (z.size(0),), device=self.device)
                
                noisy_z, noise_used = self.scheduler.add_noise(z, timesteps)
                # to accomoatde for classifier free guidance.
                noisy_z = noisy_z.repeat(2, 1, 1, 1)
                
                # this is the context embedding that the model should move towards
                context = diffusion_model.condition_embedding(path_lengths)

                # something like the a negative prompt.
                unconditional_guidance_embeddings = diffusion_model.condition_embedding(zeros_tensor)
                conditional_guidance_embeddings = torch.cat([unconditional_guidance_embeddings, context])
                

                # noise_pred = diffusion_model(noisy_z, context, timesteps)
                noise_pred = diffusion_model(noisy_z, conditional_guidance_embeddings, timesteps)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # loss = F.mse_loss(noise_pred, noise_used)
                loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise_used)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * images.size(0)
                count += images.size(0)
            
            avg_loss = epoch_loss / count if count > 0 else 0.0
            train_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        diffusion_model.eval()

        return diffusion_model, train_losses
    
    def eval_model(self, diffusion_model: Diffusion, decoder: VAE_Decoder, test_dataset: MazeTensorDataset, sample_idx: int=None, num_steps: int=50):
        self.scheduler.set_inference_timesteps(num_steps)
        
        if sample_idx is None:
            sample_idx = random.randint(0, len(test_dataset) - 1)
        
        test_img, test_path_length = test_dataset[sample_idx]
        
        context = diffusion_model.condition_embedding(
            torch.tensor([test_path_length], device=self.device).float()
        )
        
        latent = torch.randn((1, 4, 8, 8), device=self.device)
        
        for t in self.scheduler.timesteps:
            timestep = torch.tensor([t], device=self.device)
            with torch.no_grad():
                pred = diffusion_model(latent, context, timestep)
            latent = self.scheduler.step(t, latent, pred)
        
        with torch.no_grad():
            generated_image = decoder(latent / 0.18215)
        
        return generated_image, test_img, test_path_length

    def save_model(self, diffusion_model: Diffusion, path: str):
        torch.save(diffusion_model.state_dict(), path)