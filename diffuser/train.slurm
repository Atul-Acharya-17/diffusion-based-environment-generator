#!/bin/bash
#SBATCH --job-name=maze_diffusion_train     # Name of your job
#SBATCH --output=logs/train_%j.out          # Output log (%j = job ID)
#SBATCH --error=logs/train_%j.err           # Error log
#SBATCH --time=48:00:00                      # Max time (hh:mm:ss)
#SBATCH --mail-type=END,FAIL                 # Email notifications
#SBATCH --mail-user=e1373809@u.nus.edu
#SBATCH --gres=gpu:a100-40:1
#SBATCH -C cuda80

echo "Job started on $(hostname) at $(date)"

# Load environment
source ~/.bashrc
conda activate jlab || { echo "Failed to activate conda"; exit 1; }

# Print GPU info
echo "========== GPU Info =========="
nvidia-smi
echo "=============================="

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

echo "25k images, 100 epochs, f1 loss"

# Run training
python training.py 2>&1 || { echo "training.py crashed with above error"; exit 1; }
# papermill experiments_multi_feat.ipynb outs.ipynb

echo "Job finished at $(date)"
