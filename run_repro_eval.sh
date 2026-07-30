#!/bin/bash
#SBATCH --job-name=repro_eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=repro_eval_%j.log

# Stage 2b+3 ONLY: regenerate the super-resolution predictions and run THEIR
# evaluation (KL / W1 on E, omega, dissipation -- comparable to their Table 1).
#
# Does NOT retrain: train_SR_driver.py already completed in job 5824117 and its
# checkpoint is on disk. That job died only in stage 3, on an import.
#
# Needs a GPU even though the evaluation itself is cheap: SR_simulation.py names
# its output by DEVICE -- 'sr_ts<N>.pt' on CPU (:92) but 'gpu_sr_ts<N>.pt' on GPU
# (:95) -- and evaluate_sr_model.py only ever looks for the gpu_ form (:61,:64).
# Regenerating on a login node produces a file the evaluation cannot find.
#
#   sbatch run_repro_eval.sh
set -u

module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model/paper_repro

# Their Analysis/ scripts must RUN FROM Analysis/ (paths inside are relative to
# it, e.g. '../generate_data/torch_dataset/SR_test_128.pt'), but run that way
# sys.path[0] is Analysis/ and `import dataset_utils` -- which actually lives in
# generate_data/ -- fails. Absolute PYTHONPATH satisfies both.
export PYTHONPATH="$PWD/generate_data:$PWD:$PYTHONPATH"

# FAIL FAST on the inconsistency that already cost a 2 h job: their code ships
# SR_simulation.py at num_timesteps=150 but Analysis/evaluate_sr_model.py at 100,
# and both build the filename as 'gpu_sr_ts<N>.pt'. Check before spending the GPU.
TS_SIM=$(grep -oE '^num_timesteps *= *[0-9]+' SR_simulation.py | tail -1 | grep -oE '[0-9]+')
TS_EVAL=$(grep -oE '^num_timesteps *= *[0-9]+' Analysis/evaluate_sr_model.py | tail -1 | grep -oE '[0-9]+')
echo "num_timesteps -- SR_simulation.py: ${TS_SIM:-?}   evaluate_sr_model.py: ${TS_EVAL:-?}"
if [ -z "$TS_SIM" ] || [ -z "$TS_EVAL" ] || [ "$TS_SIM" != "$TS_EVAL" ]; then
    echo "ABORT: the two scripts disagree (or the constant could not be read)."
    echo "  Make them match, e.g. set SR_simulation.py to 100 by flipping its"
    echo "  lines 63/64 so '# num_timesteps = 100' becomes active:"
    echo "    sed -i '63s/^# //; 64s/^/# /' SR_simulation.py"
    exit 2
fi

STATUS=0
# Skip the regeneration if that exact file is already there.
if [ -f "predictions/gpu_sr_ts${TS_SIM}.pt" ]; then
    echo "predictions/gpu_sr_ts${TS_SIM}.pt exists -- skipping SR_simulation.py"
else
    python SR_simulation.py || STATUS=$?
fi
[ "$STATUS" -eq 0 ] && { (cd Analysis && python evaluate_sr_model.py) || STATUS=$?; }

NTFY_TOPIC="kaya-si-7h3k9x"
[ "$STATUS" -eq 0 ] && RESULT="OK" || RESULT="FAILED (exit $STATUS)"
MSG="repro_eval job ${SLURM_JOB_ID:-local} -> $RESULT"
echo "$MSG  $(date)" > "../done_repro_eval_${SLURM_JOB_ID:-local}.flag"
curl -s -m 15 -H "Title: repro_eval $RESULT" -d "$MSG" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
exit $STATUS
