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
#   sbatch run_train_si.sh                          # fixed u3232, defaults (2000 ep, stride 4)
#   sbatch run_train_si.sh 4000 2                   # EPOCHS FRAME_STRIDE (fresh)
#   sbatch run_train_si.sh 2000 4 resume            # RESUME from the checkpoint 
#   sbatch run_train_si.sh 2000 4 fresh learned     # train WITH learned physics conditioning
#   sbatch run_train_si.sh 2000 4 fresh none blind  # BLIND training (Option 1): random
#                                                   # degradations (sensor count+locations)
#
# Positionals: EPOCHS  FRAME_STRIDE  (resume|fresh)  (none|learned)  (fixed|blind)
#
# NOTE: 'linear' physics guidance needs NO special training -- train normally and
#       enable it at inference: sbatch run_inference_conditional.sh si linear 0.01
#       Only 'learned' changes the architecture and therefore needs its own run.
#       'blind' trains one model robust to many low-res inputs (own checkpoint).
# ---------------------------------------------------------------------------

EPOCHS="${1:-2000}"
FRAME_STRIDE="${2:-4}"
RESUME_ARG="${3:-}"
SI_PHYSICS="${4:-none}"      # none | learned
AUG_ARG="${5:-fixed}"        # fixed | blind

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

if [ "$AUG_ARG" != "fixed" ] && [ "$AUG_ARG" != "blind" ]; then
    echo "ERROR: fifth argument must be 'fixed' or 'blind' (got '$AUG_ARG')."
    echo "  e.g. sbatch run_train_si.sh 2000 4 fresh none blind"
    exit 1
fi
if [ "$AUG_ARG" = "blind" ]; then AUGMENT=1; else AUGMENT=0; fi

# Each variant produces a differently-behaved (or differently-shaped) network,
# so keep it in its own checkpoint -- never confuse them at inference.
CKPT=./pretrained_weights/si_ckpt
[ "$SI_PHYSICS" = "learned" ] && CKPT="${CKPT}_learned"
[ "$AUG_ARG" = "blind" ]      && CKPT="${CKPT}_blind"
CKPT="${CKPT}.pth"

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
    --si_augment "$AUGMENT" \
    --si_aug_families sensor \
    --si_aug_nmin 256 \
    --si_aug_nmax 4000 \
    --si_ckpt "$CKPT"
