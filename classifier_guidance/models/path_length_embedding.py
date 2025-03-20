import torch
from torch import nn

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
    
class EmbeddingPathLengths():
    def __init__(self):
        self.condition_embedding = ConditioningEmbedding()

    def get_path_length_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.condition_embedding(x)
    
    def get_inverse_path_length_embedding(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        x_inverse = 1 / (x + epsilon)
        return self.condition_embedding(x_inverse)
    
    def get_exp_path_length_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = torch.exp(x)
        return self.condition_embedding(x_exp)
    
    def get_log_path_length_embedding(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-8
        x_log = torch.log(x + epsilon)
        return self.condition_embedding(x_log)
    
    # need to get mean and variance for the entire dataset
    def normalize_path_lengths(self, path_lengths: torch.Tensor) -> torch.Tensor:
        normal_path_lengths = (path_lengths - path_lengths.mean()) / path_lengths.std()
        return self.condition_embedding(normal_path_lengths)