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
#   sbatch run_inference_conditional.sh dps             # with posterior sampling (DPS), zeta=0.1
#   sbatch run_inference_conditional.sh dps 0.3         # with DPS, custom zeta
#
# MODE=dps      -> routes main.py to the DPS PosteriorRunner (--run_dps 1)
# MODE=baseline -> default repository flow, reconstruct() (physics-guided)
# ---------------------------------------------------------------------------

MODE="${1:-baseline}"
ZETA="${2:-0.1}"

module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model

export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0

if [ "$MODE" = "dps" ]; then
    echo "Running WITH posterior sampling (DPS), zeta=$ZETA"
    EXTRA_ARGS="--run_dps 1 --operator sparse --zeta $ZETA"
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
