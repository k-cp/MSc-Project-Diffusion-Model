#!/bin/bash
#SBATCH --job-name=fluid_infer
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=infer_conditional_%j.log
#SBATCH --mail-user=kaya.araiyokoi@gmail.com
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch run_inference_conditional.sh                    # baseline (no posterior sampling)
#   sbatch run_inference_conditional.sh baseline           # same as above (explicit)
#   sbatch run_inference_conditional.sh dps                # posterior sampling (DPS), zeta=3.0
#   sbatch run_inference_conditional.sh dps 0.3            # DPS, custom zeta
#   sbatch run_inference_conditional.sh si                 # Stochastic Interpolants, no physics
#   sbatch run_inference_conditional.sh si linear 0.01     # SI + linear physics guidance (lambda)
#   sbatch run_inference_conditional.sh si learned 3.0     # SI + learned physics conditioning (w) from ε̃_θ = ε_θ(x_τi, τi, c) + w·[ ε_θ(x_τi, τi, c) − ε_θ(x_τi, τi, ∅) ]
#   sbatch run_inference_conditional.sh si none 0 blind    # run the BLIND checkpoint on the real u3232
#   sbatch run_inference_conditional.sh si none 0 blind sensor:512   # ROBUSTNESS eval on a held-out degradation
#
# SI positionals: si  (none|linear|learned)  (strength)  (plain|blind)  (eval-degradation)
#   eval-degradation: sensor:N | downsample:F | lowpass:K  -- builds x0 from the ground
#   truth so you can test the model on inputs it may never have seen. Empty = real u3232.
# MODE=dps      -> routes main.py to the DPS PosteriorRunner (--run_dps 1)
# MODE=si       -> routes main.py to the SIRunner (--run_si 1); train first with run_train_si.sh
#                  arg2 = physics guidance {none|linear|learned}, arg3 = its strength
#                  'linear'  works with a plain SI checkpoint (inference-only)
#                  'learned' NEEDS a checkpoint trained via: sbatch run_train_si.sh ... learned
# MODE=baseline -> default repository flow, reconstruct() (physics-guided)



# Optional phone/desktop push when the job finishes (works even if the cluster
# can't send email). Set to a PRIVATE, hard-to-guess topic, e.g. kaya-si-7h3k9x,
# then subscribe at ntfy.sh/<that-topic> (free app or browser). Empty = disabled.
# Anyone who knows the topic name can read/post, so keep it random.
NTFY_TOPIC="kaya-si-7h3k9x"

MODE="${1:-baseline}"
ZETA="${2:-3.0}"          # zeta=3.0 chosen from the sweep (best L2, stable)
SI_PHYSICS="${2:-none}"   # for MODE=si: none | linear | learned
SI_STRENGTH="${3:-0.0}"   # for MODE=si: lambda (linear) or w (learned)
SI_VARIANT="${4:-plain}"  # for MODE=si: plain | blind (which trained checkpoint)
SI_EVAL="${5:-}"          # for MODE=si: robustness eval degradation, e.g. sensor:512

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
    if [ "$SI_VARIANT" != "plain" ] && [ "$SI_VARIANT" != "blind" ]; then
        echo "ERROR: for MODE=si the fourth argument must be 'plain' or 'blind' (got '$SI_VARIANT')."
        echo "  e.g. sbatch run_inference_conditional.sh si none 0 blind"
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
    # Pick the checkpoint that matches the requested variant. 'learned' and
    # 'blind' are each trained separately, so the folder name (via --si_tag) and
    # the checkpoint must both reflect it -- otherwise the run either crashes on
    # a shape mismatch or overwrites another variant's output.
    CK=si_ckpt
    [ "$SI_PHYSICS" = "learned" ] && CK="${CK}_learned"
    [ "$SI_VARIANT" = "blind" ]   && CK="${CK}_blind"
    CK="./pretrained_weights/${CK}.pth"

    SI_ARGS="--run_si 1 --si_ckpt $CK --si_steps 100"
    [ "$SI_VARIANT" = "blind" ] && SI_ARGS="$SI_ARGS --si_tag blind"
    if [ -n "$SI_EVAL" ]; then
        echo "  robustness eval: feeding the model a '$SI_EVAL' degradation of the ground truth"
        SI_ARGS="$SI_ARGS --si_eval_degradation $SI_EVAL"
    fi

    if [ "$SI_PHYSICS" = "linear" ]; then
        echo "Running SI ($SI_VARIANT) + LINEAR physics guidance, lambda=$SI_STRENGTH  [ckpt $CK]"
        SI_ARGS="$SI_ARGS --si_physics linear --si_lambda $SI_STRENGTH"
    elif [ "$SI_PHYSICS" = "learned" ]; then
        echo "Running SI ($SI_VARIANT) + LEARNED physics conditioning, w=$SI_STRENGTH  [ckpt $CK]"
        SI_ARGS="$SI_ARGS --si_physics learned --si_w $SI_STRENGTH"
    else
        echo "Running SI ($SI_VARIANT) super-resolution (no physics guidance)  [ckpt $CK]"
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
STATUS=$?

# ---------------------------------------------------------------------------
# Completion notification. Always drops a .flag file (works everywhere); pushes
# to ntfy too if NTFY_TOPIC is set and the compute node has outbound internet.
# ---------------------------------------------------------------------------
[ "$STATUS" -eq 0 ] && RESULT="OK" || RESULT="FAILED (exit $STATUS)"
LABEL="${MODE}${SI_EVAL:+ $SI_EVAL}${SI_VARIANT:+ ($SI_VARIANT)}"
MSG="fluid_infer job ${SLURM_JOB_ID:-local}: $LABEL -> $RESULT"

echo "$MSG  $(date)" > "done_${SLURM_JOB_ID:-local}.flag"
echo "$MSG"

if [ -n "$NTFY_TOPIC" ]; then
    if curl -s -m 15 -H "Title: fluid_infer $RESULT" -d "$MSG" \
            "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1; then
        echo "notified ntfy topic '$NTFY_TOPIC'"
    else
        echo "ntfy notify failed (no outbound internet on the compute node?)"
    fi
fi

exit $STATUS
