"""Sensor-noise robustness: how fast does each method fall apart as the
measurement gets worse?

Every other result in this project rests on measurements that are EXACTLY right
(u3232 equals the ground truth to 0.0 at the sensor points), so nothing else
tests the property that decides whether a method is deployable. This reads the
--meas_noise sweep and reports both the absolute numbers and, more usefully, the
DEGRADATION RATIO relative to each method's own noise-free run -- the slope is
the robustness result, not the raw values.

Run the sweep first:
    ONLY=noise ./submit_sweep.sh
then:
    python noise_robustness.py
"""

import os
import re
import numpy as np

from metrics import normalize, spec_to_folder, _load_full, spec_to_sample_file

SIGMAS = [0.0, 0.02, 0.05]

# label -> the spec for that method at a given sigma
METHODS = {
    "baseline": lambda mn: {"method": "baseline", "mn": mn},
    "DPS":      lambda mn: {"method": "dps", "value": 3.0, "mn": mn},
    "SI":       lambda mn: {"method": "si", "mn": mn},
}


def mean_residual(folder):
    """Mean NS residual from the LAST run-block (logs accumulate on re-runs)."""
    p = os.path.join(folder, "logging_info.txt")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    for marker in ("Loading reconstruction data for DPS", "Start sampling",
                   "Loaded SI drift network"):
        if marker in txt:
            txt = marker + txt.rsplit(marker, 1)[1]
            break
    v = re.findall(r"Residual (?:final|it\d+):\s*([0-9.eE+-]+)", txt)
    return float(np.mean([float(x) for x in v])) if v else None


def load_run(spec_fn, mn, ref):
    spec = normalize(spec_fn(mn))
    folder = spec_to_folder(spec)
    if not os.path.isdir(folder):
        return None
    x = _load_full(folder, spec_to_sample_file(spec))
    if x is None:
        return None
    return {"mse": float(((x - ref) ** 2).mean()),
            "std": float(x.std()),
            "res": mean_residual(folder)}


def main():
    # reference frames come from any run that exists; they are identical everywhere
    ref = None
    for fn in METHODS.values():
        d = spec_to_folder(normalize(fn(0.0)))
        if os.path.isdir(d):
            ref = _load_full(d, "reference_arr.npy")
            break
    if ref is None:
        print("No noise-free run found; nothing to compare against.")
        return
    rstd = float(ref.std())

    print(f"{'method':10s} {'sigma':>6s} {'MSE':>9s} {'vs sigma=0':>11s} "
          f"{'std/ref%':>9s} {'residual':>10s}")
    print("-" * 60)
    for name, fn in METHODS.items():
        base = None
        for mn in SIGMAS:
            r = load_run(fn, mn, ref)
            if r is None:
                print(f"{name:10s} {mn:6.2f} {'--':>9s} {'--':>11s} {'--':>9s} {'--':>10s}")
                continue
            if mn == 0.0:
                base = r["mse"]
            ratio = f"{r['mse']/base:10.2f}x" if base else f"{'--':>11s}"
            res = f"{r['res']:10.1f}" if r["res"] is not None else f"{'--':>10s}"
            print(f"{name:10s} {mn:6.2f} {r['mse']:9.4f} {ratio} "
                  f"{100*r['std']/rstd:9.1f} {res}")
        print()

    print("'vs sigma=0' is the number that matters: it is each method's OWN")
    print("degradation, so it is not confounded by them starting at different MSE.")
    print("A method that barely moves is robust to instrument error; one that")
    print("climbs steeply relies on its measurements being right.")


if __name__ == "__main__":
    main()
