#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.log

export CUDA_VISIBLE_DEVICES=0

# Run the training script using our corrected configuration file
python main.py --config configs/kmflow_re1000_rs256.yml --exp ./experiments/km256/ --doc ./weights/km256/ --ni
