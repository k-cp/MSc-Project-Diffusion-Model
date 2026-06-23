#!/bin/bash
#SBATCH --job-name=train_diffusion
#SBATCH --partition=gpu          # (Change this to whatever partition your other script used)
#SBATCH --nodes=1
#SBATCH --gpus=1                 # Request 1 GPU
#SBATCH --mem=32G                # Request plenty of RAM for your DataLoader
#SBATCH --time=12:00:00          # Max time limit
#SBATCH --output=train_%j.log    # Save output to this log file

export CUDA_VISIBLE_DEVICES=0;

python main.py --config configs/kmflow_re1000_rs256.yml --exp ./experiments/km256/ --doc ./weights/km256/ --ni
