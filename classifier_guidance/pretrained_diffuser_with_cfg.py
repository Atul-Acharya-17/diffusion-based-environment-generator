import torch
import torch.nn as nn
from tqdm import tqdm

from classifier_guidance.models.diffusion_model import DiffusionScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model: any, scheduler: DiffusionScheduler, clip_model: any, optimizer: any, dataloader: torch.utils.data.DataLoader, guidance_scale: float = 7.5, epochs: int = 5):
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x0, _ in progress_bar:
            x0 = x0.to(device)
            t = torch.randint(0, len(scheduler.timesteps), (x0.shape[0],), device=device)

            # Encode text conditions
            text = "MNIST digit"
            text_embeddings = clip_model.text_encode(text).to(device)
            empty_embeddings = clip_model.text_encode("").to(device)
            text_embeddings = torch.cat([empty_embeddings, text_embeddings])  # [unconditional, conditional]

            # Generate noisy input
            noise = torch.randn_like(x0)
            xt = noise  # Start training from random noise

            # Predict noise for both unconditional and conditional embeddings
            noise_pred = model(xt, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # Split into uncond & cond

            # Classifier-Free Guidance Scaling
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Loss and optimization
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

    model.eval()
    return model

def sample(model: nn.Module, scheduler: DiffusionScheduler, clip_model: any, num_samples: int = 4):
    x = torch.randn((num_samples, 4, 64, 64), device=device)  # Stable Diffusion uses 4 channels
    text_embeddings = clip_model.text_encode("MNIST digit").to(device)
    empty_embeddings = clip_model.text_encode("").to(device)
    text_embeddings = torch.cat([empty_embeddings, text_embeddings])  # Concatenated for guidance

    for t in tqdm(reversed(scheduler.timesteps), desc="Sampling"):
        with torch.no_grad():
            noise_pred = model(x, torch.tensor([t], device=device), encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  # Split into uncond & cond
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            x = scheduler.step(noise_pred, t, x).prev_sample

    return x

if __name__ == "__main__":
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import matplotlib.pyplot as plt
    from diffusers import UNet2DConditionModel

    from classifier_guidance.models.clip_model import FakeCLIPModel

    pretrained_model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    clip_model = FakeCLIPModel()
    guidance_scale = 7.5
    optimizer = optim.Adam(pretrained_model.parameters(), lr=1e-3)

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = train(pretrained_model, DiffusionScheduler(timesteps=100), clip_model, optimizer, dataloader, guidance_scale)
    num_samples = 4
    x = sample(model, DiffusionScheduler(timesteps=100), clip_model, num_samples)

    x = x.clamp(-1, 1).cpu().detach().numpy() * 0.5 + 0.5
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(x[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.show()
