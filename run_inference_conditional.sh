#!/bin/bash
#SBATCH --job-name=infer_cond
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=infer_cond_%j.log

module load cray-python/3.11.7

# Activate virtual environment
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate

#Ensure we are in the main directory where kmflow_re1000_rs256.yml lives
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model


python main.py --config kmflow_re1000_rs256_conditional.yml --seed 1234 --sample_step 1 --t 240 --r 30 --ni
