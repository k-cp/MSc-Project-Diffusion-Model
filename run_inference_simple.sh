#!/bin/bash
#SBATCH --job-name=fluid_inference
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=inference_run_%j.log

# 1. Load the python module
module load cray-python/3.11.7

# Activate virtual environment
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate

#Ensure we are in the main directory where kmflow_re1000_rs256.yml lives
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model

export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0

python main.py --config kmflow_re1000_rs256.yml --seed 1234 --sample_step 1 --t 240 --r 30
