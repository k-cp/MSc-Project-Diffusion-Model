"""Figure: zero-shot generalisation across Reynolds number, on two axes.

Reads whatever cross_re* runs are present and SKIPS the rest, so it can be run
before every transfer has landed and re-run afterwards to fill the gaps in.

TWO CORRECTIONS BAKED IN, both of which change the numbers:

1. NORMALISE BY EACH Re's OWN REFERENCE STD. The logs' `mean l2 loss` is per-frame
   RMSE in absolute vorticity units, and the reference std GROWS with Re
   (4.0414 / 4.3133 / 4.4568 / 4.7081 for Re 500/1k/2k/10k). So a third of the
   apparent degradation across Re is just the field having more variance to get
   wrong. Plotting L2/std removes that confound.

2. READ EACH METHOD AGAINST ITS OWN Re1000, not against the others. The
   kf_vort_Re*_N256 family is NOT the generator the models trained on (its Re1000
   reference std is 4.3133 vs the main experiment's 4.763, a 9.4% gap), so
   absolute scores here are not comparable to the published numbers. Only the
   TREND is meaningful, and only within a method.

    python cross_re_plot.py
"""

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_ke_spectrum

RES = [500, 1000, 2000, 10000]
BASE = "guided_recons_u3232_t400_r20_w0.0"          # also the reference source per Re
METHODS = [
    ("Diffusion baseline", "#2f6ea8", BASE),
    ("DPS ($\\zeta$=3.0)", "#7a5296", "dps_guided_recons_u3232_t400_r20_w0.0_z3.0"),
    ("Stochastic interp.", "#b04a2f", "si_guided_recons_u3232_t400_r20_w0.0"),
]
OUT = "cross_re_trend.png"


def folder(re_, sub):
    return os.path.join("experiments", f"cross_re{re_}", sub)


def log_l2(d):
    """Mean L2 from the LAST run-block. Baseline/DPS write 'mean l2 loss',
    SI writes 'Mean L2 loss' -- a grep for one form silently misses the other."""
    p = os.path.join(d, "logging_info.txt")
    if not os.path.exists(p):
        return None
    txt = open(p, errors="ignore").read()
    v = re.findall(r"[Mm]ean [lL]2 loss: ([\d.]+)", txt)
    return float(v[-1]) if v else None


def mean_ek(d, fn):
    """Mean spectrum over the per-batch representative fields, or None."""
    files = sorted(glob.glob(os.path.join(d, "sample_batch*", fn)))
    if not files:
        return None, None
    tot, k = None, None
    for f in files:
        a = np.load(f)
        if a.ndim == 3:
            a = a[-1]
        k, ek = compute_ke_spectrum(a)
        tot = ek if tot is None else tot + ek
    return k, tot / len(files)


def main():
    ref_std, ref_ek = {}, {}
    for r in RES:
        d = folder(r, BASE)
        p = os.path.join(d, "sample_batch0", "reference_arr.npy")
        if not os.path.exists(p):
            print(f"  Re{r}: no reference_arr -- skipping this Re entirely")
            continue
        arrs = [np.load(f) for f in
                sorted(glob.glob(os.path.join(d, "sample_batch*", "reference_arr.npy")))]
        ref_std[r] = float(np.concatenate(arrs).std())
        _, ref_ek[r] = mean_ek(d, "reference_arr.npy")
        print(f"  Re{r}: reference std={ref_std[r]:.4f}  ({len(arrs)} batches)")

    data = {}
    for label, colour, sub in METHODS:
        for r in RES:
            if r not in ref_std:
                continue
            d = folder(r, sub)
            l2 = log_l2(d)
            if l2 is None:
                print(f"  {label} Re{r}: no log -- skipped")
                continue
            _, e = mean_ek(d, "sample_arr_run_0_it0.npy")
            if e is None:
                print(f"  {label} Re{r}: log present but NO sample arrays -- L2 only")
                espec = None
            else:
                espec = float(np.abs(e - ref_ek[r]).sum() / ref_ek[r].sum())
            data[(label, r)] = {"l2n": l2 / ref_std[r], "espec": espec}
            print(f"  {label:22s} Re{r:<6} L2/std={l2/ref_std[r]:.4f}"
                  + (f"  e_spec={espec:.3f}" if espec is not None else "  e_spec=--"))

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    for ax, key, ylab, title in (
            (axes[0], "l2n", "field error / reference std",
             "Field accuracy degrades with Re"),
            # e_spec turns out to be FLAT in Re for all three methods -- the two
            # axes fail differently, which is the point of showing both.
            (axes[1], "espec", "$e_{spec}$ — integrated spectral error",
             "Spectral fidelity does not — but SI loses its edge")):
        for label, colour, _ in METHODS:
            xs = [r for r in RES if (label, r) in data and data[(label, r)][key] is not None]
            ys = [data[(label, r)][key] for r in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", ms=7, lw=1.8, color=colour, label=label,
                    markeredgecolor="white", markeredgewidth=1.4)
            # mark any Re where this method has no point, so gaps are visible
            for r in RES:
                if r in ref_std and ((label, r) not in data
                                     or data[(label, r)][key] is None):
                    ax.plot(r, np.interp(np.log10(r), np.log10(xs), ys),
                            marker="x", ms=7, color=colour, alpha=0.45, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(RES)
        ax.set_xticklabels([str(r) for r in RES])
        ax.set_xlabel("Reynolds number (zero-shot; models trained at Re=1000)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11.5, loc="left")
        ax.grid(alpha=0.22)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Zero-shot generalisation across Reynolds number",
                 fontsize=13, x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.012,
             "Field error is normalised by each Re's own reference std (4.04 → 4.76 across the "
             "range), so the trend is not the reference's variance. SI's e_spec is 0.226 even at "
             "Re=1000 — against 0.008 on the benchmark — so the spectral collapse is the change "
             "of generator, not of Reynolds number.",
             fontsize=8.5, color="#5d6b7c")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(OUT, dpi=180)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
