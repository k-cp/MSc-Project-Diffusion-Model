#!/bin/bash
#SBATCH --job-name=train_si
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=train_si_%j.log

# Memory is auto-allocated (one GPU = a full GH200 superchip), so no --mem.
# Max walltime on Isambard is 24h; to train longer, resubmit with resume=1.

# ---------------------------------------------------------------------------
# Train the Stochastic Interpolant drift network. Produces
# ./pretrained_weights/si_ckpt.pth, which is then used by:
#   sbatch run_inference_conditional.sh si
#
# Usage:
#   sbatch run_train_si.sh                 # fresh, defaults (2000 epochs, frame_stride 4)
#   sbatch run_train_si.sh 4000 2          # EPOCHS FRAME_STRIDE (fresh)
#   sbatch run_train_si.sh 2000 4 resume   # RESUME from si_ckpt.pth (after a timeout)
# ---------------------------------------------------------------------------

EPOCHS="${1:-2000}"
FRAME_STRIDE="${2:-4}"
RESUME_ARG="${3:-}"

if [ "$RESUME_ARG" = "resume" ]; then
    RESUME=1
else
    RESUME=0
fi

module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model

export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0

python train_si.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --epochs "$EPOCHS" \
    --batch_size 32 \
    --lr 2e-4 \
    --frame_stride "$FRAME_STRIDE" \
    --resume "$RESUME" \
    --si_ckpt ./pretrained_weights/si_ckpt.pth
