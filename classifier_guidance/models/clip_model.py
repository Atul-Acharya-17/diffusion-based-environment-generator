import torch

class FakeCLIPModel:
    def text_encode(self, text):
        """Mocked CLIP encoder: Maps text to a pseudo embedding."""
        return torch.tensor([len(text) % 100 / 100], dtype=torch.float32).unsqueeze(0)