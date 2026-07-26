"""Summarize the SI robustness experiment into one table.

After running the eval sweep -- the SAME held-out degradations on the specialist
and the blind checkpoint -- this reads each run's log and tabulates final L2 and
physics residual, so you can see how each model degrades as the input drifts from
what it was trained on.

Run the sweep first, e.g.:
    for D in sensor:256 sensor:512 sensor:1024 sensor:2048 downsample:8 lowpass:4; do
        sbatch run_inference_conditional.sh si none 0 plain  $D     # specialist
        sbatch run_inference_conditional.sh si none 0 blind  $D     # blind
    done
then:  python si_robustness.py
"""

import os
import re
import numpy as np

EXPERIMENT_FOLDER = "kmflow_re1000_rs256_ddim_conditional_new"
BASE = "si_guided_recons_u3232_t400_r20_w0.0"

# Which trained checkpoints to compare (column label -> folder tag suffix).
VARIANTS = {"specialist": "", "blind": "_blind"}

# Held-out degradations to sweep (rows). "" = the real u3232 sensor file.
DEGRADATIONS = ["", "sensor:256", "sensor:512", "sensor:1024",
                "sensor:2048", "downsample:8", "lowpass:4"]


def eval_folder(tag, degradation):
    name = BASE + tag
    if degradation:
        name += "_eval_" + degradation.replace(":", "")
    return os.path.join("experiments", EXPERIMENT_FOLDER, name)


def read_metrics(folder):
    """Return (mean_L2, mean_residual) from a run's log, or (None, None)."""
    log = os.path.join(folder, "logging_info.txt")
    if not os.path.exists(log):
        return None, None
    txt = open(log, errors="ignore").read()
    # A folder can be re-run, appending fresh blocks to the same log. Each run
    # starts by loading the drift network, so only average residuals from the
    # LAST such block -- otherwise a stale/broken earlier run poisons the mean.
    marker = "Loaded SI drift network"
    if marker in txt:
        txt = marker + txt.rsplit(marker, 1)[1]
    l2 = re.findall(r"Mean L2 loss:\s*([0-9.eE+-]+)", txt)
    res = re.findall(r"Residual final:\s*([0-9.eE+-]+)", txt)
    # The "Mean L2 loss" summary is printed only when a run finishes. Its absence
    # means the run is still in progress (or was cut off), so the residuals cover
    # only a partial, unrepresentative slice of the test set -- skip it entirely
    # rather than report a biased half-run average.
    if not l2:
        return None, None
    mean_l2 = float(l2[-1])
    mean_res = float(np.mean([float(x) for x in res])) if res else None
    return mean_l2, mean_res


def main():
    variants = list(VARIANTS)
    # header
    hdr = f"{'degradation':16s} | " + " | ".join(f"{v+' L2':>14s}" for v in variants) \
          + " || " + " | ".join(f"{v+' resid':>14s}" for v in variants)
    print(hdr)
    print("-" * len(hdr))

    for deg in DEGRADATIONS:
        label = deg if deg else "u3232 (real)"
        l2s, ress = [], []
        for v in variants:
            l2, res = read_metrics(eval_folder(VARIANTS[v], deg))
            l2s.append(l2)
            ress.append(res)
        fmt = lambda x: f"{x:14.4f}" if isinstance(x, float) else f"{'--':>14s}"
        row = f"{label:16s} | " + " | ".join(fmt(x) for x in l2s) \
              + " || " + " | ".join(fmt(x) for x in ress)
        print(row)

    print("\nL2 = mean field error vs ground truth (lower better).")
    print("resid = mean Navier-Stokes residual (lower = more physical).")
    print("'--' = run not found (submit that (variant, degradation) job first).")
    print("\nReading the table: the SPECIALIST should be best on u3232 and get")
    print("progressively worse off-distribution; the BLIND model should stay flatter")
    print("across degradations -- that gap IS the robustness result.")


if __name__ == "__main__":
    main()
