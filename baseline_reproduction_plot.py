"""Figure: the baseline reproduction, and what it reveals about the NS residual.

Retraining the baseline from scratch reproduces the provided checkpoint to within
0.2% on MSE, corr, std, e_spec, slope, KE and enstrophy -- and the three networks
agree with EACH OTHER 80x more closely than any of them agrees with the truth
(pairwise MSE 0.044-0.046 vs 3.63-3.65 against ground truth, pairwise corr 0.9986).
So the baseline's error is a property of the METHOD, not of a particular
checkpoint: independent training runs make the same mistakes. That forecloses the
obvious objection to the headline result -- the baseline was not simply unlucky.

The one metric that does NOT reproduce is the residual: 62.84 -> 84.99 (+35%),
and the median moves with the mean (61.78 -> 81.13), so it is a uniform shift
rather than a few outlier batches. A field 1.3% different yields a residual 35%
different -- roughly 27x amplification -- because the residual applies derivative
operators (~k^2) and magnifies exactly the small-scale content that MSE and
e_spec barely register. That is the mechanism behind the residual's unreliability
everywhere else in this project.

    python baseline_reproduction_plot.py
"""

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_ke_spectrum

E = "experiments/kmflow_re1000_rs256_ddim_conditional_new"
RUNS = [
    ("provided weights", "#2f6ea8", f"{E}/guided_recons_u3232_t400_r20_w0.0"),
    ("retrained here",   "#7a5296", f"{E}/guided_recons_u3232_t400_r20_w0.0_mine"),
    ("retrained + x-shift", "#2c7a5c", f"{E}/guided_recons_u3232_t400_r20_w0.0_mine_xshift"),
]
OUT = "baseline_reproduction.png"


def per_batch_residual(folder):
    txt = open(os.path.join(folder, "logging_info.txt"), errors="ignore").read()
    if "Start sampling" in txt:
        txt = txt.rsplit("Start sampling", 1)[1]
    return np.array([float(x) for x in
                     re.findall(r"Residual it\d+:\s*([0-9.eE+-]+)", txt)])


def mean_ek(folder, fn="sample_arr_run_0_it0.npy"):
    tot, k = None, None
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", fn)))
    for f in files:
        a = np.load(f)
        if a.ndim == 3:
            a = a[-1]
        k, ek = compute_ke_spectrum(a)
        tot = ek if tot is None else tot + ek
    return k, tot / len(files)


def main():
    k, e_ref = mean_ek(RUNS[0][2], "reference_arr.npy")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # --- (a) per-batch residual: the ONE metric that does not reproduce --------
    ax = axes[0]
    for i, (label, colour, d) in enumerate(RUNS):
        r = per_batch_residual(d)
        if r.size == 0:
            continue
        x = np.random.default_rng(0).normal(i, 0.055, r.size)   # jitter for visibility
        ax.plot(x, r, "o", ms=4.5, color=colour, alpha=0.42, markeredgewidth=0)
        ax.plot([i - 0.26, i + 0.26], [r.mean()] * 2, color=colour, lw=2.4, zorder=3)
        ax.text(i + 0.31, r.mean(), f"mean {r.mean():.1f}\nmedian {np.median(r):.1f}",
                fontsize=8.5, color=colour, va="center")
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels(["provided", "retrained", "retrained\n+ x-shift"], fontsize=9.5)
    ax.set_xlim(-0.45, len(RUNS) - 0.05)
    ax.set_ylabel("NS residual, per batch")
    ax.set_title("The residual does not reproduce (+35%)", fontsize=11.5, loc="left")
    ax.grid(axis="y", alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- (b) WHY: they differ only in a band that carries no energy ----------
    # Log y, because the interesting divergence is a factor of ~1.4 sitting on
    # top of a ratio that runs from 0.4 to 37 -- linear axes hide everything
    # below k=100 entirely.
    ax = axes[1]
    ax.axvspan(80, k.max(), color="#9a7218", alpha=0.09, zorder=0)
    ax.axhline(1.0, color="#8c96a3", lw=1.1, ls="--", zorder=1)
    for label, colour, d in RUNS:
        _, e = mean_ek(d)
        ax.plot(k, e / e_ref, lw=1.7, color=colour, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel("$E(k) / E_{ref}(k)$")
    ax.set_title("…and differ only where the energy is ~0", fontsize=11.5, loc="left")
    ax.annotate("shaded: $k>80$ — holds 0.0000% of the total\n"
                "energy, but the residual weights it by $k^2$",
                xy=(0.03, 0.045), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.5, color="#7a5a18")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.22, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Retraining the baseline reproduces it — except on the residual",
                 fontsize=13, x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.026,
             "The three agree with each other 80× more closely than any agrees with truth "
             "(pairwise MSE 0.044–0.046 vs 3.63–3.65; corr 0.9986).",
             fontsize=8.5, color="#5d6b7c")
    fig.text(0.008, 0.006,
             "At k=127 the ratio is 26.2 / 37.1 / 26.4 — which orders the residuals exactly "
             "(62.8 / 85.0 / 63.9). Energy-based metrics are blind to it; the residual is not.",
             fontsize=8.5, color="#5d6b7c")
    fig.tight_layout(rect=(0, 0.062, 1, 0.96))
    fig.savefig(OUT, dpi=180)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
