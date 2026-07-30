"""Figure: a low Navier-Stokes residual does NOT mean a faithful spectrum.

The residual is the physics metric this project quotes most, and on its own it
is misleading: several very different configurations reach a low residual by
OVER-SMOOTHING -- removing the high-k content that the residual's derivative
operators punish, along with the real turbulence living there. Ground truth's
own residual is ~12.5, so anything far below it is suspect by construction.

Plotting every run as (residual, e_spec) shows the two axes are close to
uncorrelated: the runs nearest the truth on residual are frequently among the
worst on integrated spectral error. e_spec comes from metrics.deployability_
metrics' own definition, computed with metrics.compute_ke_spectrum unmodified,
so these values match the numbers quoted elsewhere.

    python residual_vs_spectrum_plot.py
"""

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import (EXPERIMENT_FOLDER, _mean_ek, compute_ke_spectrum,
                     normalize, spec_to_folder)

OUT = "residual_vs_spectrum.png"
TRUTH_RESIDUAL = 12.5          # ground truth's own residual -- the honest target

# Runs worth naming. Everything else is drawn as an unlabelled point so the
# CLOUD carries the argument and the labels only mark its corners.
HIGHLIGHT = {
    "guided_recons_u3232_t400_r20_w0.0":            ("baseline", "#2f6ea8"),
    "guided_recons_u3232_t400_r100_w0.0":           ("baseline r=100", "#2f6ea8"),
    "guided_recons_u3232_t400_r20_w0.0_sm7":        ("+ input smoothing", "#2f6ea8"),
    "dps_guided_recons_u3232_t400_r20_w0.0_z3.0":   ("DPS", "#7a5296"),
    "si_guided_recons_u3232_t400_r20_w0.0":         ("SI", "#b04a2f"),
    "si_guided_recons_u3232_t400_r20_w0.0_eval_lowpass4": ("SI · lowpass:4", "#b04a2f"),
}

# Label placement in AXES fraction, hand-tuned so nothing collides. Log axes make
# automatic placement unreliable, and there are only six labels.
LABEL_XY = {
    "baseline r=100":    (0.08, 0.93),
    "+ input smoothing": (0.44, 0.80),
    "baseline":          (0.60, 0.90),
    "DPS":               (0.72, 0.40),
    "SI":                (0.34, 0.16),
    "SI · lowpass:4":    (0.03, 0.42),
}


# Two folders arrived without their logging_info.txt, so their residual cannot be
# read back. Only one of them matters -- the headline SI run -- and its value is
# on record in SI_README.md:539 (§7 results table), which is the number quoted
# everywhere else in this project.
#
# Deliberately NOT recomputed from the arrays: voriticity_residual needs a 3-frame
# sequence for dw/dt, but sample_arr stores only the MIDDLE frame of each sliding
# window (slice2sequence keeps data[:,1:2]). Rebuilding triples from consecutive
# saved frames would score frames reconstructed in DIFFERENT windows, which is a
# different object from what the logs measured -- and would quietly disagree with
# every residual already in the writeup.
FALLBACK_RESIDUAL = {
    "si_guided_recons_u3232_t400_r20_w0.0": 8.28,      # SI_README.md:539
}


def family(name):
    if name.startswith("dps_"):
        return "DPS", "#7a5296"
    if name.startswith("si_"):
        return "SI", "#b04a2f"
    return "baseline", "#2f6ea8"


def mean_residual(folder):
    """Mean NS residual from the LAST run-block only.

    A reused output folder's log accumulates blocks; averaging all of them once
    inflated a residual from 8.3 to 507, so the split is not optional.
    """
    p = os.path.join(folder, "logging_info.txt")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    for marker in ("Loading reconstruction data for DPS", "Start sampling",
                   "Loaded SI drift network"):
        if marker in txt:
            txt = marker + txt.rsplit(marker, 1)[1]
            break
    v = [float(x) for x in re.findall(r"Residual (?:final|it\d+):\s*([0-9.eE+-]+)", txt)]
    return float(np.mean(v)) if v else None


def sample_file_for(folder):
    """A --sample_step N run stores its final answer in it{N-1}, not it0."""
    cands = sorted(glob.glob(os.path.join(folder, "sample_batch0", "sample_arr_run_0_it*.npy")))
    return os.path.basename(cands[-1]) if cands else None


def main():
    root = os.path.join("experiments", EXPERIMENT_FOLDER)
    ref_dir = spec_to_folder(normalize({"method": "baseline"}))
    k, e_ref = _mean_ek(ref_dir, "reference_arr.npy")
    print(f"Reference spectrum from {ref_dir}")

    pts = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or not os.path.isdir(os.path.join(d, "sample_batch0")):
            continue
        name = os.path.basename(d)
        sf = sample_file_for(d)
        if sf is None:
            print(f"  skip {name} (no sample array -- empty/aborted run folder)")
            continue
        res = mean_residual(d)
        if res is None and name in FALLBACK_RESIDUAL:
            res = FALLBACK_RESIDUAL[name]
            print(f"  {name}: no log; using recorded residual {res} (SI_README.md:539)")
        if res is None:
            print(f"  skip {name} (no logging_info.txt and no recorded value)")
            continue
        try:
            _, e = _mean_ek(d, sf)
        except FileNotFoundError:
            print(f"  skip {name} (no spectrum)")
            continue
        e_spec = float(np.abs(e - e_ref).sum() / e_ref.sum())
        pts.append({"name": name, "res": res, "e": e_spec})
        print(f"  {name:56s} residual={res:9.2f}  e_spec={e_spec:6.3f}")

    if not pts:
        print("Nothing to plot.")
        return

    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for p in pts:
        lab, col = HIGHLIGHT.get(p["name"], (None, family(p["name"])[1]))
        if lab:
            ax.plot(p["res"], p["e"], marker="o", ms=10, color=col,
                    markeredgecolor="white", markeredgewidth=1.6, zorder=4)
        else:
            ax.plot(p["res"], p["e"], marker="o", ms=5.5, color=col,
                    alpha=0.32, markeredgewidth=0, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.9, 9000)
    ax.set_ylim(0.0035, 1.15)          # headroom so labels clear the top row

    # ground truth's own residual -- the reference every "low residual" claim
    # should be read against
    ax.axvline(TRUTH_RESIDUAL, color="#2c7a5c", lw=1.3, ls="--", zorder=1)
    ax.annotate("ground truth (12.5)", xy=(TRUTH_RESIDUAL, 1.0),
                xytext=(4, 0), textcoords="offset points",
                color="#2c7a5c", fontsize=9, va="top", ha="left", rotation=90)

    for p in pts:
        lab, col = HIGHLIGHT.get(p["name"], (None, None))
        if not lab:
            continue
        ax.annotate(lab, xy=(p["res"], p["e"]),
                    xytext=LABEL_XY.get(lab, (0.5, 0.5)), textcoords="axes fraction",
                    fontsize=9.5, color=col, weight="medium",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, alpha=0.55,
                                    shrinkA=2, shrinkB=6))

    ax.set_xlabel("mean Navier–Stokes residual  (log scale)")
    ax.set_ylabel("$e_{spec}$  — integrated relative spectral error  (log scale)")
    ax.set_title("A low residual is not a faithful spectrum",
                 fontsize=13, loc="left", pad=14)
    ax.grid(alpha=0.2, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.text(0.012, 0.028,
             f"{len(pts)} runs; faint points are the remaining configurations. Down is a "
             "better spectrum. The residual axis has no 'good' direction:",
             fontsize=8.5, color="#5d6b7c")
    fig.text(0.012, 0.006,
             "truth sits at 12.5, and both sides of it are wrong — to the right by injecting "
             "high-k noise, to the left by smoothing the turbulence away.",
             fontsize=8.5, color="#5d6b7c")

    fig.tight_layout(rect=(0, 0.058, 1, 1))
    fig.savefig(OUT, dpi=180)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
