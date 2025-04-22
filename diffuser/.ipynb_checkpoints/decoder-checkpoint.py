# decoder
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape

        x = x.view(n, c, h * w)
        x = x.transpose(-1, -2)

        x = self.attention(x)
        x = x.transpose(-1, -2)
        
        x = x.view((n, c, h, w))

        x += residue
        return x
        
class VAE_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, In_channels, H, W)

        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # Input projection
            # nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # nn.Conv2d(4, 256, kernel_size=3, padding=1),
            
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            nn.Conv2d(8, 256, kernel_size=3, padding=1),
            
            # Residual blocks
            VAE_ResBlock(256, 256),
            VAE_AttnBlock(256),
            VAE_ResBlock(256, 256),
            
            # First upsampling (8x8 -> 16x16)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            VAE_ResBlock(128, 128),
            VAE_ResBlock(128, 128),
            
            # Second upsampling (16x16 -> 32x32)
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            VAE_ResBlock(64, 64),
            VAE_ResBlock(64, 64),

            # Additional padding to achieve 10x10 output 
            # nn.ConstantPad2d((1, 1, 1, 1), 0),
            
            # Output projection
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH.SIZE, 4, H / 8, W / 8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # x = x[:, :, 1:-1, 1:-1]

        return x
