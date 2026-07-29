#!/bin/bash
# ---------------------------------------------------------------------------
# EXACT reproduction of Schiodt et al. (Sci. Rep. 2026) -- THEIR code, THEIR
# solver, THEIR ConvNeXt architecture, THEIR training config and metrics.
# This script clones their public repository and remaps the two hardcoded
# path prefixes (the authors' DTU home + external drive) into the clone.
#
#   ./setup_paper_repro.sh [target_dir]     # default: ./paper_repro
#
# Then, on the cluster:
#   sbatch run_repro_datagen.sh     # ~5 h CPU: their DNS -> 2500 snapshots + datasets
#   sbatch run_repro_train.sh       # GPU: their SI_full training + SR + evaluation
#
# LAPTOP SHORTCUT: the raw snapshots can be generated locally (numpy-only,
# ~4 h, ~656 MB) and scp'd into <target>/data/ -- then skip the datagen job's
# solver stage (it detects existing snapshots and only assembles the .pt sets).
#
# The reproduction is EXACT in method (code, architecture, hyperparameters,
# metrics) and STATISTICAL in data: their RNG seeds produce their realisations
# of the flow; a rerun produces new realisations of the same distribution, so
# reproduced numbers should match their Table 1 to within sampling error.
# ---------------------------------------------------------------------------
set -e
TARGET="${1:-paper_repro}"
git clone https://github.com/martinschiodt/Turbulence_Stochastic_Interpolants.git "$TARGET"
cd "$TARGET"
ROOT="$(pwd)"
mkdir -p data models predictions generate_data/torch_dataset

# remap the authors' hardcoded prefixes into this clone
grep -rl "/zhome/80/1/88013/Code/Python/TurbulenceEnrichment" --include="*.py" . \
  | xargs sed -i.bak "s#/zhome/80/1/88013/Code/Python/TurbulenceEnrichment#${ROOT}#g"
grep -rl "/media/martin/Tracers/KolmogorovFlow2D" --include="*.py" . \
  | xargs sed -i.bak "s#/media/martin/Tracers/KolmogorovFlow2D/Omega_hat/omega_hat#${ROOT}/data/omega_hat#g; s#/media/martin/Tracers/KolmogorovFlow2D/case128/omega_hat_#${ROOT}/data/omega_hat_#g"
find . -name "*.bak" -delete
echo "Patched. Data -> $ROOT/data, models -> $ROOT/models, predictions -> $ROOT/predictions"
