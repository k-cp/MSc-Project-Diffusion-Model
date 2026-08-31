#!/usr/bin/env python3
"""Emit the two curves behind the scalar physical metrics: the in-void energy
spectrum and the in-void vorticity distribution.

    python make_physics_figure.py            # -> Thesis/figures/physics_si_vs_unet.pdf

WHY THIS FIGURE EXISTS. tab_stats reports std ratio, enstrophy ratio and KL, and
each of those is a whole curve collapsed to one number. eval_si_inpaint.py says
why that is not enough, in its own words: "one number cannot say WHICH scales
were lost or WHICH part of the vorticity tail was clipped, and those are the
questions a fluids reader asks." The left panel answers the first, the right
panel the second.

IT COVERS EVERY LEARNED METHOD AT THE DEEPEST CONFIGURATION, which is where the
scalars disagree most. The regression U-Net comes out ahead of the interpolant on
variance, enstrophy and KL -- the opposite of what regression to the conditional
mean is supposed to do -- and a scalar cannot separate "genuinely sharper" from
"high-wavenumber noise". A spectrum can, and that is the difference between a
finding and an artefact.

BOTH PANELS ARE IN-VOID. The whole-field versions are diluted by the pixels
copied verbatim from the truth, and the dilution factor is the coverage
(eval_si_inpaint.py, module docstring). Plotting whole-field curves here would
make every method look close to correct for a reason that has nothing to do with
the method.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

NPZ = "figures/metrics_si_inpaint_depth.npz"
NPZ_BL = "figures/metrics_baselines_depth.npz"
OUT = "Thesis/figures/physics_in_void.pdf"

# Every method at the deepest configuration whose curves exist. THE FOUR
# CLASSICAL ROWS OF tab:stats ARE ABSENT AND THAT IS A TOOLING GAP, NOT A CHOICE:
# baselines_inpaint.py computes the in-void spectrum (it pops ek_p_in / ek_r_in to
# form e_spec_in) and then saves only the depth profiles, so the curve is thrown
# away exactly as eval_si_inpaint.py used to throw its own away before 2026-08-13.
# Persisting it there and re-running fills these panels in; until then the figure
# covers the learned methods and the caption says so rather than implying the
# classical fills were left out on purpose.
METHODS = [
    ("zero fill",           "zero_single_n1_g25",              "#95a5a6", (0, (1, 3))),
    ("harmonic",            "harmonic_single_n1_g25",          "#16a085", (0, (4, 2))),
    ("biharmonic",          "biharmonic_single_n1_g25",        "#8e44ad", (0, (2, 2))),
    ("gappy POD",           "gpod512_single_n1_g25",           "#d68910", (0, (5, 1, 1, 1))),
    ("FNO",                 "single_n1_g25_rand_fno",          "#7f8c8d", ":"),
    ("interpolant",         "single_n1_g25_rand_noema_noxshift", "#1f4e79", "-"),
    ("interpolant, centre", "single_n1_g25_ctr_noema_noxshift", "#2e86c1", "-."),
    ("U-Net (regression)",  "single_n1_g25_ctr_unet",          "#c0392b", "--"),
]

# TWO FILES, TWO SEPARATORS. eval_si_inpaint.py writes "<run>__<key>" and
# baselines_inpaint.py writes "<run>|<key>". Aligning them would invalidate every
# existing reader of either file, so this merges on read instead.
def load(path, sep):
    try:
        z = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        return {}
    return {k.replace(sep, "\x00"): z[k] for k in z.files}

z = {}
z.update(load(NPZ, "__"))
z.update(load(NPZ_BL, "|"))
get = lambda run, key: z.get(run + "\x00" + key)
# Methods whose curves are not on disk are DROPPED WITH A PRINTED WARNING rather
# than silently omitted -- a figure that quietly shows four of eight methods is the
# failure this whole codebase keeps guarding against.
have, absent = [], []
for m in METHODS:
    (have if get(m[1], "ek_pred_void") is not None else absent).append(m)
if absent:
    print("  NOT PLOTTED (no saved curves): " + ", ".join(m[0] for m in absent))
    print("  -> re-run baselines_inpaint.py; it now persists ek_pred_void/ek_ref_void.")
if not have:
    sys.exit("no curves at all -- is the npz the refreshed copy?")
METHODS = have

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.5))

# --- left: the in-void spectra themselves ------------------------------------
# RAW CURVES, NOT A RATIO. \citet{shu2023physics} Fig. 7 plots each method and the
# reference as separate curves, and a reader holding the two figures side by side
# should not have to convert between presentations. The ratio version answered
# "how much energy is missing at each scale" more directly, but it has no
# counterpart in the literature this chapter is arguing with.
# THIS IS A VORTICITY SPECTRUM, NOT A KINETIC ENERGY ONE. The dataset is vorticity
# and every array in the eval is vorticity, so labelling it E(k) as a kinetic
# energy spectrum -- which is what the JCP paper plots, from velocity -- would be
# wrong. The two are not interchangeable and the axis says which this is.
ref = get(METHODS[0][1], "ek_ref_void")
axL.loglog(np.arange(1, len(ref) + 1), ref, color="0.15", lw=1.8, label="truth")
for label, key, colour, style in METHODS:
    p = np.asarray(get(key, "ek_pred_void"), float)
    axL.loglog(np.arange(1, len(p) + 1), p, ls=style, color=colour, lw=1.4, label=label)
axL.set_xlabel("wavenumber $k$ (in-void)")
axL.set_ylabel("$E_\\omega(k)$, vorticity spectrum")
axL.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2)

# --- right: the in-void vorticity distribution -------------------------------
# Log density, because the whole question is the TAIL. On a linear axis the two
# methods and the truth sit on top of one another in the core and the extreme
# events -- the part a fluids reader cares about -- are invisible.
edges = np.asarray(get(METHODS[0][1], "hist_edges"), float)
centres = 0.5 * (edges[:-1] + edges[1:])
dx = float(edges[1] - edges[0])


def density(a):
    a = np.asarray(a, float)
    s = a.sum() * dx
    return a / s if s > 0 else a


axR.semilogy(centres, density(get(METHODS[0][1], "hist_ref")),
             color="0.25", lw=1.4, label="truth")
for label, key, colour, style in METHODS:
    h = get(key, "hist_pred")
    if h is None:
        continue
    axR.semilogy(centres, density(h), ls=style, color=colour, lw=1.4, label=label)
axR.set_xlabel("vorticity $\\omega$ (in-void)")
# $p(\omega)$, matching \citet{shu2023physics} Fig. 7b. Their axis is LINEAR and
# ours is log -- see the note above; the label is aligned, the scale deliberately
# is not.
axR.set_ylabel("$p(\\omega)$")
axR.set_xlim(-25, 25)
axR.set_ylim(1e-6, 1)
axR.legend(frameon=False, fontsize=6.5, ncol=2)

for ax in (axL, axR):
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200)
print("wrote", OUT)

# The numbers behind the picture, so a striking curve can be checked against the
# table rather than taken on trust.
for label, key, _, _ in METHODS:
    p = np.asarray(get(key, "ek_pred_void"), float)
    r = np.asarray(get(key, "ek_ref_void"), float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r > 0, p / r, np.nan)
    print("  %-20s worst-k energy ratio %.3f at k=%d;  high-k (k>32) mean %.3f"
          % (label, np.nanmin(ratio), int(np.nanargmin(ratio)) + 1,
             np.nanmean(ratio[32:])))
