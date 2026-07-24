"""JCP-style comparison table: L2 norm and Equation loss per method.

Mirrors Table 2 of Shu, Li & Farimani (JCP 2023): rows = methods, columns =
  * L2 norm       -- the repo's l2_loss (per-frame RMSE vs ground truth, averaged),
                     recomputed here from the saved arrays (exact, always current).
  * Equation loss -- the mean Navier-Stokes vorticity residual. This needs 3
                     consecutive frames, which the saved (middle-frame) arrays don't
                     carry, so it is read from each run's logging_info.txt (the value
                     the runner computed with the real physics). Uses the LAST run
                     block in the log, so stale earlier runs don't pollute it.

Selection uses the same spec system as metrics.py, so it works with EVERY config
(baseline / dps / si, physics, blind, eval). Edit METHODS at the bottom.
"""

import os
import re
import glob
import numpy as np

from metrics import normalize, spec_to_folder, spec_to_label, EXPERIMENT_FOLDER


def _load_full(folder, file_name):
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", file_name)))
    if not files:
        return None
    return np.concatenate([np.load(f) for f in files]).astype(np.float64)


def l2_norm(folder):
    """Repo l2_loss: per-frame RMSE vs ground truth, then averaged."""
    s = _load_full(folder, "sample_arr_run_0_it0.npy")
    r = _load_full(folder, "reference_arr.npy")
    if s is None or r is None:
        return None
    return float(np.sqrt(((s - r) ** 2).mean(axis=(-1, -2))).mean())


def equation_loss(folder):
    """Mean NS residual from the run's log (last run block)."""
    log = os.path.join(folder, "logging_info.txt")
    if not os.path.exists(log):
        return None
    txt = open(log, errors="ignore").read()
    for marker in ("Loading reconstruction data for", "Doing sparse reconstruction task"):
        if marker in txt:
            txt = txt.split(marker)[-1]        # keep only the last run
            break
    m = re.findall(r"mean residual loss:\s*([0-9.eE+-]+)", txt)   # baseline summary
    if m:
        return float(m[-1])
    vals = re.findall(r"Residual final:\s*([0-9.eE+-]+)", txt)    # SI / DPS per-batch
    if not vals:
        vals = re.findall(r"Residual it\d+:\s*([0-9.eE+-]+)", txt)
    return float(np.mean([float(v) for v in vals])) if vals else None


def compile_table(methods):
    rows = []
    for entry in methods:
        try:
            spec = normalize(entry)
        except ValueError as e:
            print(f"Skipping {entry!r}: {e}")
            continue
        folder = spec_to_folder(spec)
        if not os.path.isdir(folder):
            print(f"Warning: folder not found, skipping: {folder}")
            continue
        rows.append((spec_to_label(spec), l2_norm(folder), equation_loss(folder)))

    if not rows:
        print("No valid runs found.")
        return

    def fmt(x, w, p):
        return f"{x:{w}.{p}f}" if isinstance(x, float) else f"{'--':>{w}s}"

    print(f"\n{'Method':32s} {'L2 norm':>10s} {'Equation loss':>15s}")
    print("-" * 60)
    lines = ["| Method | L2 norm | Equation loss |", "|---|---|---|"]
    for label, l2, eq in rows:
        print(f"{label:32s} {fmt(l2, 10, 4)} {fmt(eq, 15, 3)}")
        lines.append(f"| {label} | {fmt(l2, 0, 4).strip()} | {fmt(eq, 0, 3).strip()} |")

    print("\nL2 norm  = per-frame RMSE vs ground truth (lower = closer).")
    print("Equation loss = mean Navier-Stokes residual (lower = more physical); "
          "from the run log.")
    print("'--' equation loss = no log in the folder (copy logging_info.txt, or re-run).")

    out = os.path.join("experiments", EXPERIMENT_FOLDER, "metrics_table_jcp.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"\nSaved (markdown): {out}")


if __name__ == "__main__":
    # Same entry style as metrics.py: shorthand, (shorthand, value), or dict spec.
    METHODS = ["baseline", ("dps", 3.0), "si", ("si_linear", 0.01), "si_blind"]
    compile_table(METHODS)
