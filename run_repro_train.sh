#!/bin/bash
#SBATCH --job-name=repro_train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=repro_train_%j.log

# Stage 2: THEIR SI_full training (ConvNeXt drift, batch 40, 4000 epochs),
# then THEIR super-resolution inference and evaluation -- output directly
# comparable to their Table 1 (KL / W1 on E, omega, dissipation).
module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model/paper_repro

# Their Analysis/ scripts must run FROM Analysis/ -- the paths inside them are
# relative to it ('../generate_data/torch_dataset/SR_test_128.pt'). But run that
# way, sys.path[0] is Analysis/, so `import dataset_utils` (which actually lives
# in generate_data/) fails. Fix both at once: cd into Analysis so the relative
# data paths resolve, and put the helper dirs on PYTHONPATH as ABSOLUTE paths so
# the cd doesn't break them.
export PYTHONPATH="$PWD/generate_data:$PWD:$PYTHONPATH"

# Skip stages whose output already exists, so a failure in a later stage never
# costs a re-run of the ~2 h training.
[ -f predictions/gpu_sr_ts150.pt ] || { python train_SR_driver.py && python SR_simulation.py; }
STATUS=$?
[ "$STATUS" -eq 0 ] && { (cd Analysis && python evaluate_sr_model.py); STATUS=$?; }

NTFY_TOPIC="kaya-si-7h3k9x"
[ "$STATUS" -eq 0 ] && RESULT="OK" || RESULT="FAILED (exit $STATUS)"
MSG="repro_train job ${SLURM_JOB_ID:-local} -> $RESULT"
echo "$MSG  $(date)" > "../done_repro_train_${SLURM_JOB_ID:-local}.flag"
curl -s -m 15 -H "Title: repro_train $RESULT" -d "$MSG" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
exit $STATUS
