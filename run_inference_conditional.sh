#!/bin/bash

#SBATCH --job-name=fluid_dps_inference
#SBATCH --output=logs/dps_%j.out
#SBATCH --error=logs/dps_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00


source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_env

mkdir -p logs

python main.py \
    --config configs/kmflow_re1000_rs256.yml \
    --sample_step 2 \
    --scale_factor 4 \
    --zeta 0.5 \
    --comment "Running Diffusion Posterior Sampling reconstruction baseline"