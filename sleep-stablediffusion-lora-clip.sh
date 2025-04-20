#!/bin/sh
#SBATCH --job-name=diffusion_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1503329@comp.nus.edu.sg
#SBATCH --time=48:00:00                     # Max time (hh:mm:ss)
#SBATCH --gres=gpu:a100-40:1
#SBATCH -C cuda80

# Execute the Jupyter notebook file non-interactively
srun jupyter nbconvert --execute --to notebook --inplace ./diffuser/StableDiffusion_LoRA_CLIP.ipynb

# scp -r ./sleep-stablediffusion-lora.sh e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code
# scp -r ./StableDiffusion_LoRA-CLIP.ipynb e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/diffuser
# scp -r ./StableDiffusion.ipynb e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/diffuser
# scp -r ./diffusion-based-environment-generator/generator e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code
# scp -r ./diffusion-based-environment-generator/requirements.txt e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/requirements.txt
# scp -r e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/loss_curves_20250420_204426/stable_diffusion_diffusion_models/diffusion_image_generation_multi_feat_20250420_204426.png .
# scp -r e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/loss_curves_20250420_204426/stable_diffusion_diffusion_models/loss_curve_diffusion_multi_feat_20250420_204426.png .
# scp -r ./diffusion-based-environment-generator/data/vae_weights_20250326_065408.pth e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/vae_weights_20250326_06540