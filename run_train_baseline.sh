#!/bin/bash
#SBATCH --job-name=train_baseline
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=train_baseline_%j.log

# Memory is auto-allocated (one GPU = a full GH200 superchip), so no --mem.
# Max walltime on Isambard is 24h; the conditional model is ~12h for 300 epochs.

# ---------------------------------------------------------------------------
# Retrain the BASELINE conditional diffusion model from scratch, so the headline
# comparison rests on weights whose training we control rather than the
# checkpoint shipped with the repo (whose training details we can't inspect).
#
# Same architecture and config as the provided model, and the SAME data split:
# KMFlowTensorDataset(train_ratio=0.9) trains on the first 36 of 40 trajectories
# and holds out the last 4 -- exactly the frames inference is scored on, and the
# same split train_si.py uses. So baseline/SI/DPS stay apples-to-apples.
#
# Usage:
#   sbatch run_train_baseline.sh              # fresh run, 300 epochs (config default)
#   sbatch run_train_baseline.sh resume       # continue from the last checkpoint
#
# AFTER TRAINING -- publish the weights under the name inference expects:
#   cp train_ddpm/experiments/km256_mine/logs/weights/km256_mine/ckpt.pth \
#      pretrained_weights/conditional_ckpt_mine.pth
# then run it:
#   sbatch run_inference_conditional.sh baseline mine
# and compare against the provided weights:
#   sbatch run_inference_conditional.sh baseline given
#
# Checkpoint format note: training saves [model, optim, epoch, step] and appends
# the EMA weights last when model.ema is true (it is). Inference loads index
# [-1], i.e. the EMA weights -- so this checkpoint drops straight in, no surgery.
# ---------------------------------------------------------------------------

RESUME_ARG="${1:-fresh}"

if [ "$RESUME_ARG" != "fresh" ] && [ "$RESUME_ARG" != "resume" ]; then
    echo "ERROR: first argument must be 'fresh' or 'resume' (got '$RESUME_ARG')."
    echo "  e.g. sbatch run_train_baseline.sh resume"
    exit 1
fi

module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model/train_ddpm

export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0

# --ni answers "delete existing logs?" non-interactively. On a FRESH run that
# wipes the old folder, which is what we want; on a resume we must NOT pass it,
# and --resume_training picks up from ckpt.pth instead.
if [ "$RESUME_ARG" = "resume" ]; then
    echo "RESUMING baseline training from the last checkpoint"
    MODE_ARGS="--resume_training"
else
    echo "Training baseline conditional diffusion model FRESH (overwrites previous run)"
    MODE_ARGS="--ni"
fi

python main.py \
    --config km_re1000_rs256_conditional.yml \
    --exp ./experiments/km256_mine/ \
    --doc ./weights/km256_mine/ \
    --seed 1234 \
    $MODE_ARGS

echo "Training finished. Publish the weights with:"
echo "  cp train_ddpm/experiments/km256_mine/logs/weights/km256_mine/ckpt.pth pretrained_weights/conditional_ckpt_mine.pth"
