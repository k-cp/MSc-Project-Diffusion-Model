#!/bin/bash
#SBATCH --job-name=repro_datagen
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=repro_datagen_%j.log

# Stage 1 of the Schiodt et al. reproduction: run THEIR pseudo-spectral DNS
# (Re=1000, 128^2, dt=0.0025, samples every 1000 steps for decorrelation):
# 15k spin-up steps, then 2500 snapshots; then assemble their .pt datasets.
# If data/omega_hat_2499.npy already exists (e.g. scp'd from the laptop run),
# the solver stage is skipped and only the assembly runs.
module load cray-python/3.11.7
source /scratch/u6ki/kayaay.u6ki/diffusion_env/bin/activate
cd /scratch/u6ki/kayaay.u6ki/MSc-Project-Diffusion-Model/paper_repro

if [ ! -f data/omega_hat_2499.npy ]; then
    cd generate_data
    python -c "
import re, pathlib
s = pathlib.Path('KolmogorovFlow2d.py').read_text()
pathlib.Path('_solver_init.py').write_text(s.replace(\"sim_type = 'Run'\", \"sim_type = 'Init'\"))
"
    python _solver_init.py          # 15k spin-up -> omega_hat_0.npy
    python KolmogorovFlow2d.py      # 2500 decorrelated snapshots (~4-5 h)
    cd ..
fi
cd generate_data
python assemble_superresolution_dataset.py   # -> SR_{train,valid,test}_128.pt
STATUS=$?
cd ..

NTFY_TOPIC="kaya-si-7h3k9x"
[ "$STATUS" -eq 0 ] && RESULT="OK" || RESULT="FAILED (exit $STATUS)"
MSG="repro_datagen job ${SLURM_JOB_ID:-local} -> $RESULT"
echo "$MSG  $(date)" > "../done_repro_datagen_${SLURM_JOB_ID:-local}.flag"
curl -s -m 15 -H "Title: repro_datagen $RESULT" -d "$MSG" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1
exit $STATUS
