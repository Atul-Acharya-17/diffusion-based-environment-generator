import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, t, encoder_hidden_states=None):
        x = self.encoder(x)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.view(-1, 1, 1, 1).expand_as(x)
            x = x + encoder_hidden_states
        x = self.decoder(x)
        return x
    
class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = list(range(timesteps))
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def step(self, noise_pred, t, x):
        """Reverse step: Estimate x_{t-1} from x_t using the predicted noise"""
        x = x - noise_pred / len(self.timesteps)
        return type('Dummy', (object,), {'prev_sample': x})