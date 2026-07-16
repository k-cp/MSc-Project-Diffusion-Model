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
#   sbatch run_train_si.sh                    # fresh, defaults (2000 epochs, frame_stride 4)
#   sbatch run_train_si.sh 4000 2             # EPOCHS FRAME_STRIDE (fresh)
#   sbatch run_train_si.sh 2000 4 resume      # RESUME from si_ckpt.pth (after a timeout)
#   sbatch run_train_si.sh 2000 4 fresh learned   # train WITH learned physics conditioning
#
# NOTE: 'linear' physics guidance needs NO special training -- train normally and
#       enable it at inference: sbatch run_inference_conditional.sh si linear 0.01
#       Only 'learned' changes the architecture and therefore needs its own run.
# ---------------------------------------------------------------------------

EPOCHS="${1:-2000}"
FRAME_STRIDE="${2:-4}"
RESUME_ARG="${3:-}"
SI_PHYSICS="${4:-none}"      # none | learned

if [ "$RESUME_ARG" = "resume" ]; then
    RESUME=1
else
    RESUME=0
fi

if [ "$SI_PHYSICS" != "none" ] && [ "$SI_PHYSICS" != "learned" ]; then
    echo "ERROR: fourth argument must be 'none' or 'learned' (got '$SI_PHYSICS')."
    echo "  e.g. sbatch run_train_si.sh 2000 4 fresh learned"
    exit 1
fi

# 'learned' produces an architecturally different network -> keep it in its own
# checkpoint so it can never be confused with the plain SI one.
if [ "$SI_PHYSICS" = "learned" ]; then
    CKPT=./pretrained_weights/si_ckpt_learned.pth
else
    CKPT=./pretrained_weights/si_ckpt.pth
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
    --si_physics "$SI_PHYSICS" \
    --si_ckpt "$CKPT"
