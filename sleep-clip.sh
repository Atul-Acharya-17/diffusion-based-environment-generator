#!/bin/sh
#SBATCH --job-name=diffusion_test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1503329@comp.nus.edu.sg
#SBATCH --time=48:00:00                     # Max time (hh:mm:ss)
#SBATCH --gres=gpu:a100-40:1
#SBATCH -C cuda80

# Execute the Jupyter notebook file non-interactively
srun jupyter nbconvert --execute --to notebook --inplace ./diffuser/experiments_multi_feat_stablediffusion_vae_clip.ipynb

# scp -r ./sleep.sh e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code
# scp -r ./sleep-clip.sh e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code
# scp -r ./experiments_multi_feat_stablediffusion_vae.ipynb e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/diffuser
# scp -r ./experiments_multi_feat_stablediffusion_vae_clip.ipynb e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/diffuser
# scp -r ./diffusion-based-environment-generator/generator e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code
# scp -r ./diffusion-based-environment-generator/requirements.txt e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/requirements.txt
# scp -r e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/loss_curves_20250331_012340/loss_curve_diffusion_multi_feat_20250331_012340.png .
# scp -r e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/loss_curves_20250421_230124/stable_diffusion_diffusion_models/diffusion_image_generation_multi_feat_20250421_230124.png .
# scp -r e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/loss_curves_20250421_230124/stable_diffusion_diffusion_models/loss_curve_diffusion_multi_feat_20250421_230124.png .
# scp -r ./diffusion-based-environment-generator/data/vae_weights_20250326_065408.pth e1503329@xlogin.comp.nus.edu.sg:/home/e/e1503329/abhishek/diffusion_generator_code/data/vae_weights_20250326_065408.pth