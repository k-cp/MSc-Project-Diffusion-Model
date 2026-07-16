#!/bin/bash
#SBATCH --job-name=fluid_infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=infer_conditional_%j.log


# Usage:
#   sbatch run_inference_conditional.sh                    # baseline (no posterior sampling)
#   sbatch run_inference_conditional.sh baseline           # same as above (explicit)
#   sbatch run_inference_conditional.sh dps                # posterior sampling (DPS), zeta=3.0
#   sbatch run_inference_conditional.sh dps 0.3            # DPS, custom zeta
#   sbatch run_inference_conditional.sh si                 # Stochastic Interpolants, no physics
#   sbatch run_inference_conditional.sh si linear 0.01     # SI + linear physics guidance (lambda)
#   sbatch run_inference_conditional.sh si learned 3.0     # SI + learned physics conditioning (w)
#
# MODE=dps      -> routes main.py to the DPS PosteriorRunner (--run_dps 1)
# MODE=si       -> routes main.py to the SIRunner (--run_si 1); train first with run_train_si.sh
#                  arg2 = physics guidance {none|linear|learned}, arg3 = its strength
#                  'linear'  works with a plain SI checkpoint (inference-only)
#                  'learned' NEEDS a checkpoint trained via: sbatch run_train_si.sh ... learned
# MODE=baseline -> default repository flow, reconstruct() (physics-guided)



MODE="${1:-baseline}"
ZETA="${2:-3.0}"          # zeta=3.0 chosen from the sweep (best L2, stable)
SI_PHYSICS="${2:-none}"   # for MODE=si: none | linear | learned
SI_STRENGTH="${3:-0.0}"   # for MODE=si: lambda (linear) or w (learned)

# Fail loudly on a bad mode instead of silently running the baseline.
# (Common mistake: `sbatch run_inference_conditional.sh 0.1` -- the first arg
# is the MODE, not zeta. Correct: `... dps 0.1`.)
if [ "$MODE" != "baseline" ] && [ "$MODE" != "dps" ] && [ "$MODE" != "si" ]; then
    echo "ERROR: first argument must be 'baseline', 'dps' or 'si' (got '$MODE')."
    echo "Usage: sbatch run_inference_conditional.sh [baseline|dps|si] [zeta | physics strength]"
    echo "  baseline:  sbatch run_inference_conditional.sh"
    echo "  DPS:       sbatch run_inference_conditional.sh dps 0.3"
    echo "  SI:        sbatch run_inference_conditional.sh si"
    echo "  SI+phys:   sbatch run_inference_conditional.sh si linear 0.01"
    echo "             sbatch run_inference_conditional.sh si learned 3.0"
    exit 1
fi

# Same idea for the SI physics mode: reject a bad value instead of silently
# running without guidance.
if [ "$MODE" = "si" ]; then
    if [ "$SI_PHYSICS" != "none" ] && [ "$SI_PHYSICS" != "linear" ] && [ "$SI_PHYSICS" != "learned" ]; then
        echo "ERROR: for MODE=si the second argument must be 'none', 'linear' or 'learned' (got '$SI_PHYSICS')."
        echo "  e.g. sbatch run_inference_conditional.sh si linear 0.01"
        exit 1
    fi
fi

module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model

export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0

if [ "$MODE" = "dps" ]; then
    echo "Running WITH posterior sampling (DPS), zeta=$ZETA"
    EXTRA_ARGS="--run_dps 1 --operator sparse --zeta $ZETA"
elif [ "$MODE" = "si" ]; then
    SI_ARGS="--run_si 1 --si_ckpt ./pretrained_weights/si_ckpt.pth --si_steps 100"
    if [ "$SI_PHYSICS" = "linear" ]; then
        echo "Running Stochastic Interpolants + LINEAR physics guidance, lambda=$SI_STRENGTH"
        SI_ARGS="$SI_ARGS --si_physics linear --si_lambda $SI_STRENGTH"
    elif [ "$SI_PHYSICS" = "learned" ]; then
        echo "Running Stochastic Interpolants + LEARNED physics conditioning, w=$SI_STRENGTH"
        SI_ARGS="$SI_ARGS --si_physics learned --si_w $SI_STRENGTH"
    else
        echo "Running Stochastic Interpolant super-resolution (no physics guidance)"
    fi
    EXTRA_ARGS="$SI_ARGS"
else
    echo "Running WITHOUT posterior sampling (baseline reconstruct)"
    EXTRA_ARGS=""
fi

python main.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --t 400 \
    --r 20 \
    $EXTRA_ARGS
