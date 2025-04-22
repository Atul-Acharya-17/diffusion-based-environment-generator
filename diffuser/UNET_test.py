import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_maze_from_test(sample_idx=None, num_steps=50):
    diffusion_model.eval()
    scheduler.set_inference_timesteps(num_steps)
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(test_dataset) - 1)
    
    test_img, test_path_length = test_dataset[sample_idx]
    
    context = diffusion_model.condition_embedding(
        torch.tensor([test_path_length], device=device).float()
    )
    
    latent = torch.randn((1, 4, 8, 8), device=device)
    
    for t in scheduler.timesteps:
        timestep = torch.tensor([t], device=device)
        with torch.no_grad():
            pred = diffusion_model(latent, context, timestep)
        latent = scheduler.step(t, latent, pred)
    
    with torch.no_grad():
        generated_image = decoder(latent / 0.18215)
    
    return generated_image, test_img, test_path_length

generated, original, test_path_length = generate_maze_from_test(num_steps=1000)

generated_np = generated.squeeze(0).permute(1, 2, 0).cpu().numpy()
generated_np = (generated_np + 1) / 2.0

original_np = original.permute(1, 2, 0).cpu().numpy()
original_np = (original_np + 1) / 2.0

generated_display = generated_np

threshold_value = 0.5
generated_approx = (generated_np > threshold_value).astype(np.float32)

block_size = 4
H, W, C = generated_approx.shape
new_H = H // block_size
new_W = W // block_size

final_blocks = np.zeros((new_H, new_W, C), dtype=np.float32)
for i in range(new_H):
    for j in range(new_W):
        block = generated_approx[i * block_size:(i + 1) * block_size,
                                  j * block_size:(j + 1) * block_size, :]
        final_blocks[i, j, :] = (block.mean(axis=(0, 1)) >= 0.5).astype(np.float32)

final_image = np.repeat(np.repeat(final_blocks, block_size, axis=0), block_size, axis=1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(generated_display)
plt.title("Generated Maze")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(generated_approx)
plt.title("Thresholded Approximation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_image)
plt.title("Final Blocky Maze")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(original_np)
plt.title(f"Original Test Maze | path len: {test_path_length}")
plt.axis("off")
plt.show()