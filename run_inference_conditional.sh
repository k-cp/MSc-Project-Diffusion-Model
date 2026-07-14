#!/bin/bash
#SBATCH --job-name=fluid_infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=infer_conditional_%j.log

# ---------------------------------------------------------------------------
# Usage:
#   sbatch run_inference_conditional.sh                 # baseline (no posterior sampling)
#   sbatch run_inference_conditional.sh baseline        # same as above (explicit)
#   sbatch run_inference_conditional.sh dps             # posterior sampling (DPS), zeta=3.0
#   sbatch run_inference_conditional.sh dps 0.3         # DPS, custom zeta
#   sbatch run_inference_conditional.sh si              # Stochastic Interpolants (needs trained ckpt)
#
# MODE=dps      -> routes main.py to the DPS PosteriorRunner (--run_dps 1)
# MODE=si       -> routes main.py to the SIRunner (--run_si 1); train first with run_train_si.sh
# MODE=baseline -> default repository flow, reconstruct() (physics-guided)
# ---------------------------------------------------------------------------

MODE="${1:-baseline}"
ZETA="${2:-3.0}"          # zeta=3.0 chosen from the sweep (best L2, stable)

# Fail loudly on a bad mode instead of silently running the baseline.
# (Common mistake: `sbatch run_inference_conditional.sh 0.1` -- the first arg
# is the MODE, not zeta. Correct: `... dps 0.1`.)
if [ "$MODE" != "baseline" ] && [ "$MODE" != "dps" ] && [ "$MODE" != "si" ]; then
    echo "ERROR: first argument must be 'baseline', 'dps' or 'si' (got '$MODE')."
    echo "Usage: sbatch run_inference_conditional.sh [baseline|dps|si] [zeta]"
    echo "  baseline:  sbatch run_inference_conditional.sh"
    echo "  DPS:       sbatch run_inference_conditional.sh dps 0.3"
    echo "  SI:        sbatch run_inference_conditional.sh si"
    exit 1
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
    echo "Running Stochastic Interpolant super-resolution"
    EXTRA_ARGS="--run_si 1 --si_ckpt ./pretrained_weights/si_ckpt.pth --si_steps 100"
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
