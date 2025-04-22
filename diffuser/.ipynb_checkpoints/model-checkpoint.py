# UNET + TIME EMEBEDDINGS
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

## embedding time
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

## temp condition embedding will be replaced with the actual embedding:

class ConditioningEmbedding(nn.Module):
    def __init__(self, d_context=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, d_context),
        )
        
    def forward(self, x):
        x = x.float()
        return self.proj(x.unsqueeze(-1)).unsqueeze(1)  # (B, 1, d_context)
        # return self.proj(x.unsqueeze(-1))  # (B, d_context)
    
class MultiFeatureConditioningEmbeddingDynamic(nn.Module): 
    def __init__(self, input_size, d_context=768): 
        super().__init__() 
        # input_size is dynamically set based on the number of features provided 
        self.proj = nn.Sequential( nn.Linear(input_size, 256), 
                                  nn.SiLU(), 
                                  nn.Linear(256, d_context), 
                                  )  
    def forward(self, x): 
        x = x.float() # Convert to float 
        return self.proj(x).unsqueeze(1) # Output shape: (B, 1, d_context)

## Unet residual Block
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_features = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.group_norm_merge = nn.GroupNorm(32, out_channels)
        self.conv_merge =  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_features(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.group_norm_merge(merged)

        return merged + self.residual_layer(residue)


class UNET_AttnBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        residue_long = x
        
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        
        x += residue_short
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


# upsample block
class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
        
# squential block
class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttnBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(nn.Conv2d(8, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttnBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttnBlock(8, 40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttnBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttnBlock(8, 80)),

            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3,stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttnBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttnBlock(8, 160)),
            
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3,stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),   
        ])

        self.bottleneck = SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttnBlock(8, 160), UNET_ResidualBlock(1280, 1280),)

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttnBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttnBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttnBlock(8, 160), UpSample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttnBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttnBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttnBlock(8, 80), UpSample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttnBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttnBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttnBlock(8, 40)),            
        ])

        
    def forward(self, x, context, time):

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x

class Diffusion(nn.Module):

    def __init__(self, input_size=1, num_train_timesteps=1000):
        ## check if correct??????
        super().__init__()
        self.time_embed = nn.Embedding(num_train_timesteps, 320)
        self.time_mlp = TimeEmbedding(320)
        self.unet = UNET()
        # self.final = UNET_OutputLayer(320, 4)
        self.final = UNET_OutputLayer(320, 8)
        self.condition_embedding = ConditioningEmbedding()
        self.condition_multidimensional_embedding = MultiFeatureConditioningEmbeddingDynamic(input_size)

    def forward(self, latent, context, timesteps):
        t_emb = self.time_mlp(self.time_embed(timesteps))
        return self.final(self.unet(latent, context, t_emb))

