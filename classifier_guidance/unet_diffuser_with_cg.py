import torch
import torch.nn as nn
from tqdm import tqdm

from classifier_guidance.models.diffusion_model import UNet, DiffusionScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model: nn.Module, scheduler: DiffusionScheduler, classifier_model: nn.Module, optimizer: any, dataloader: torch.utils.data.DataLoader, guidance_scale: float = 7.5, epochs: int = 5):
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x0, y in progress_bar:
            x0 = x0.to(device)
            t = torch.randint(0, len(scheduler.timesteps), (x0.shape[0],), device=device)

            noise = torch.randn_like(x0)
            xt = noise

            # Predict noise
            noise_pred = model(xt, t)

            y = torch.randint(0, 10, (x0.shape[0],), device=device)  # Random class labels
            class_guidance = classifier_model.get_class_guidance(xt, y)

            noise_pred += class_guidance * guidance_scale

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

    model.eval()
    return model

def sample(model: nn.Module, scheduler: DiffusionScheduler, classifier_model: nn.Module, num_samples: int = 4):
    y = torch.tensor([1] * num_samples, device=device)
    x = torch.randn((num_samples, 1, 28, 28), device=device)

    for t in tqdm(reversed(scheduler.timesteps), desc="Sampling"):
        with torch.no_grad():
            noise_pred = model(x, t)
        class_guidance = classifier_model.get_class_guidance(x, y)
        noise_pred += class_guidance * guidance_scale
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x

if __name__ == "__main__":
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    import matplotlib.pyplot as plt
    from classifier_guidance.models.classifier_model import FakeClassifier

    model = UNet().to(device)
    scheduler = DiffusionScheduler(timesteps=100)
    classifier_model = FakeClassifier().to(device)
    guidance_scale = 7.5
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = train(model, scheduler, classifier_model, optimizer, dataloader, guidance_scale)

    num_samples = 4
    x = sample(model, scheduler, classifier_model, num_samples)

    x = x.clamp(-1, 1).cpu().detach().numpy() * 0.5 + 0.5
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(x[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.show()
