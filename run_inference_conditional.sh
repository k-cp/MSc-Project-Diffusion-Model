#!/bin/bash
#SBATCH --job-name=fluid_dps
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=dps_recons_%j.log


module load cray-python/3.11.7


source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate


cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model


export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0


python main.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --run_dps 1 \
    --scale_factor 4 \
    --zeta 0.1 \
    --comment "Running customized Diffusion Posterior Sampling super resolution"