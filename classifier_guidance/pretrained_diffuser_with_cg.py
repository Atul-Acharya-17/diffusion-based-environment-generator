import torch
import torch.nn as nn
from tqdm import tqdm

from classifier_guidance.models.diffusion_model import DiffusionScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

def sample(model: nn.Module, scheduler: DiffusionScheduler, classifier_model: nn.Module, num_samples: int = 4):
    y = torch.tensor([1] * num_samples, device=device)
    x = torch.randn((num_samples, 4, 64, 64), device=device)

    for t in tqdm(reversed(scheduler.timesteps), desc="Sampling"):
        with torch.no_grad():
            noise_pred = model(x, torch.tensor([t], device=device)).sample
        class_guidance = classifier_model.get_class_guidance(x, y)
        noise_pred += class_guidance * guidance_scale
        x = scheduler.step(noise_pred, t, x).prev_sample

    return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from diffusers import UNet2DConditionModel
    from classifier_guidance.models.classifier_model import FakeClassifier

    model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    scheduler = DiffusionScheduler(timesteps=100)
    classifier_model = FakeClassifier().to(device)
    guidance_scale = 7.5

    model.eval()
    num_samples = 4

    x = sample()

    x = x.clamp(-1, 1).cpu().detach().numpy() * 0.5 + 0.5
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(x[i, 0], cmap="gray")
        axes[i].axis("off")
    plt.show()
