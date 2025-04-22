# encoder
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttnBlock, VAE_ResBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # Initial convolution
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            
            # Residual blocks
            VAE_ResBlock(64, 64),
            VAE_ResBlock(64, 64),
            
            # First downsampling (32x32 -> 16x16)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            
            # Increased channels
            VAE_ResBlock(64, 128),
            VAE_ResBlock(128, 128),
            
            # Second downsampling (16x16 -> 8x8)
            # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # Final processing
            VAE_ResBlock(128, 256),
            VAE_ResBlock(256, 256),
            VAE_AttnBlock(256),

            # nn.ConstantPad2d((0, 1, 0, 1), 0),
            
            # Output projection
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            # nn.Conv2d(256, 8, kernel_size=3, padding=1),
            # nn.Conv2d(8, 8, kernel_size=1, padding=0)
            nn.Conv2d(256, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channel, H, W)
        # noise: (Batch_size, out_channels, H/8, W/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        # (Batch_size, 8, H, H / 8, W / 8) -> 2 tensors of shape (Batch size, 4, H/8, W/8)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        x = mean + stdev * noise

        x *= 0.18215

        # return x
        return mean, log_var, x