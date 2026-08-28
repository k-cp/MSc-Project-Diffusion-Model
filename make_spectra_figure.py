#!/usr/bin/env python3
"""In-void energy spectrum, SI against the FNO, at three depths.

Uses the VOID-ONLY spectra (ek_*_void) from the evaluator's npz, not the
whole-field ones: whole-field spectra are diluted by the observed region, which
the evaluation protocol excludes from claims.

    python make_spectra_figure.py            # -> Thesis/figures/spectra_si_vs_fno.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "figures/metrics_si_inpaint_depth.npz"
OUT = "Thesis/figures/spectra_si_vs_fno.pdf"

# single-square voids only: one geometry, three depths, so depth is the only variable
PANELS = [("single_n1_g0.6_rand",  r"$0.6\%$, void $20^2$",  "10 px"),
          ("single_n1_g6.25_rand", r"$6.25\%$, void $64^2$", "32 px"),
          ("single_n1_g25_rand",   r"$25\%$, void $128^2$",  "64 px")]

SI_C, FNO_C = "#2C6E9B", "#C4761B"

z = np.load(NPZ, allow_pickle=True)
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)

for ax, (run, title, depth) in zip(axes, PANELS):
    for suffix, colour, label in [("", SI_C, "interpolant"), ("_fno", FNO_C, "FNO")]:
        pred = z[f"{run}{suffix}__ek_pred_void"]
        ref = z[f"{run}{suffix}__ek_ref_void"]
        ratio = pred / np.maximum(ref, 1e-30)
        k = np.arange(1, len(ratio) + 1)
        ax.plot(k, ratio, color=colour, lw=1.6, label=label)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("wavenumber $k$ in the void")
    ax.set_title(f"{title}\nmax depth {depth}", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

axes[0].set_ylabel(r"$E_{\mathrm{pred}}(k) \;/\; E_{\mathrm{truth}}(k)$")
axes[0].legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)

for run, title, depth in PANELS:
    line = []
    for suffix, tag in [("", "SI"), ("_fno", "FNO")]:
        r = z[f"{run}{suffix}__ek_pred_void"] / np.maximum(z[f"{run}{suffix}__ek_ref_void"], 1e-30)
        line.append(f"{tag} min {r.min():.2f}")
    print(f"  {depth:>6}: " + "   ".join(line))
