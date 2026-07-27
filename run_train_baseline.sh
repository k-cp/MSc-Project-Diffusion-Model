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
# Usage:  sbatch run_train_baseline.sh [fresh|resume] [noxshift|xshift]
#   sbatch run_train_baseline.sh                    # plain REPRODUCTION (default)
#   sbatch run_train_baseline.sh resume             # continue from the last checkpoint
#   sbatch run_train_baseline.sh fresh xshift       # + x-translation augmentation
#
# x-translation: the Kolmogorov forcing -4cos(4y) varies only along columns on a
# periodic domain, so rolling ROWS maps a solution to another exact solution --
# free, physics-valid data. (Rotations/flips would move the forcing bands and are
# NOT valid.) The provided checkpoint was trained WITHOUT it, so 'noxshift' is the
# reproduction and 'xshift' is a deliberately different, hopefully better model.
# Each variant gets its own run dir and checkpoint name, so they never collide.
#
# AFTER TRAINING -- publish the weights under the name inference expects
# (the script prints the exact command for the variant you ran):
#   cp train_ddpm/experiments/km256_mine/logs/weights/km256_mine/ckpt.pth \
#      pretrained_weights/conditional_ckpt_mine.pth
# then run it:
#   sbatch run_inference_conditional.sh baseline mine        # reproduction
#   sbatch run_inference_conditional.sh baseline mine_xshift # augmented
# and compare against the provided weights:
#   sbatch run_inference_conditional.sh baseline given
#
# Checkpoint format note: training saves [model, optim, epoch, step] and appends
# the EMA weights last when model.ema is true (it is). Inference loads index
# [-1], i.e. the EMA weights -- so this checkpoint drops straight in, no surgery.
# ---------------------------------------------------------------------------

RESUME_ARG="${1:-fresh}"
XSHIFT_ARG="${2:-noxshift}"   # noxshift | xshift

if [ "$RESUME_ARG" != "fresh" ] && [ "$RESUME_ARG" != "resume" ]; then
    echo "ERROR: first argument must be 'fresh' or 'resume' (got '$RESUME_ARG')."
    echo "  e.g. sbatch run_train_baseline.sh resume"
    exit 1
fi

if [ "$XSHIFT_ARG" != "noxshift" ] && [ "$XSHIFT_ARG" != "xshift" ]; then
    echo "ERROR: second argument must be 'noxshift' or 'xshift' (got '$XSHIFT_ARG')."
    echo "  e.g. sbatch run_train_baseline.sh fresh xshift"
    exit 1
fi

# Separate run directory per variant, so the augmented model never overwrites the
# plain reproduction. NOTE: the plain run is the one that reproduces the provided
# checkpoint -- adding augmentation changes the recipe, so it is a DIFFERENT model,
# not a reproduction. Keep both if you want to claim either.
if [ "$XSHIFT_ARG" = "xshift" ]; then
    RUN=km256_mine_xshift
    AUG_ARGS="--x_translate 1"
    CKPT_NAME=conditional_ckpt_mine_xshift.pth
else
    RUN=km256_mine
    AUG_ARGS=""
    CKPT_NAME=conditional_ckpt_mine.pth
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

echo "Run directory: $RUN   (augmentation: $XSHIFT_ARG)"

python main.py \
    --config km_re1000_rs256_conditional.yml \
    --exp ./experiments/$RUN/ \
    --doc ./weights/$RUN/ \
    --seed 1234 \
    $MODE_ARGS $AUG_ARGS
STATUS=$?

echo "Training finished. Publish the weights with:"
echo "  cp train_ddpm/experiments/$RUN/logs/weights/$RUN/ckpt.pth pretrained_weights/$CKPT_NAME"

# ---------------------------------------------------------------------------
# Completion notification, mirroring run_inference_conditional.sh. This is a
# ~12 h job, so knowing it finished (or died) without polling matters more here
# than for the 20-minute inference runs.
# ---------------------------------------------------------------------------
NTFY_TOPIC="kaya-si-7h3k9x"

[ "$STATUS" -eq 0 ] && RESULT="OK" || RESULT="FAILED (exit $STATUS)"
MSG="train_baseline job ${SLURM_JOB_ID:-local} ($RESUME_ARG/$XSHIFT_ARG) -> $RESULT"

# cd'd into train_ddpm above, so drop the flag at the repo root next to the
# inference flags rather than burying it one level down.
echo "$MSG  $(date)" > "../done_train_${SLURM_JOB_ID:-local}.flag"
echo "$MSG"

if [ -n "$NTFY_TOPIC" ]; then
    if curl -s -m 15 -H "Title: train_baseline $RESULT" -d "$MSG" \
            "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1; then
        echo "notified ntfy topic '$NTFY_TOPIC'"
    else
        echo "ntfy notify failed"
    fi
fi

exit $STATUS
