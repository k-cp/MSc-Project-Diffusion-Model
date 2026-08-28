#!/usr/bin/env python3
"""The depth law: error against local depth, before and after dividing by void size.

Left panel is the raw profiles for all nine configurations. Right panel divides
each by its void's edge length, which is the claim: at a fixed distance from
data, error is roughly proportional to how big the void is.

    python make_depth_law_figure.py     # -> Thesis/figures/depth_law.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ = "figures/metrics_si_inpaint_depth.npz"
OUT = "Thesis/figures/depth_law.pdf"

# run -> (void edge length in px, geometry)
RUNS = {
    "random_n6_g0.6_rand":     (8,   "random"),
    "random_n64_g6.25_rand":   (8,   "random"),
    "random_n256_g25_rand":    (8,   "random"),
    "multiple_n4_g0.6_rand":   (10,  "multiple"),
    "multiple_n4_g6.25_rand":  (32,  "multiple"),
    "multiple_n4_g25_rand":    (64,  "multiple"),
    "single_n1_g0.6_rand":     (20,  "single"),
    "single_n1_g6.25_rand":    (64,  "single"),
    "single_n1_g25_rand":      (128, "single"),
}
COLOUR = {"single": "#C0392B", "multiple": "#C4761B", "random": "#2C6E9B"}

z = np.load(NPZ, allow_pickle=True)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.6))

seen = set()
for run, (edge, geom) in RUNS.items():
    d = z[f"{run}__depth_centres"]
    v = z[f"{run}__depth_relL2"]
    ok = np.isfinite(v) & (v > 0)
    label = geom if geom not in seen else None
    seen.add(geom)
    ax0.plot(d[ok], v[ok], color=COLOUR[geom], lw=1.4, marker="o", ms=2.5, label=label)
    ax1.plot(d[ok], v[ok] / edge, color=COLOUR[geom], lw=1.4, marker="o", ms=2.5)

for ax, title in ((ax0, "raw"), (ax1, "divided by void edge length")):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("distance to the nearest observed pixel (px)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5, which="both")
ax0.set_ylabel(r"in-void $\mathrm{rel}L_2$")
ax1.set_ylabel(r"$\mathrm{rel}L_2$ / edge")
ax0.legend(frameon=False, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)

# the numbers the text quotes, recomputed here so figure and prose cannot drift
for D in (1, 3, 5, 7, 9):
    E, V = [], []
    for run, (edge, _) in RUNS.items():
        d, v = z[f"{run}__depth_centres"], z[f"{run}__depth_relL2"]
        i = int(np.argmin(abs(d - D)))
        if abs(d[i] - D) < 1.5 and np.isfinite(v[i]) and v[i] > 0:
            E.append(edge); V.append(v[i])
    if len(E) < 4:
        continue
    x, y = np.log(E), np.log(V)
    p, c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (p * x + c)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    col = np.array(V) / np.array(E, float) ** p
    print("  d=%2d px  n=%d  p=%.2f  R2=%.2f  spread %.1fx -> %.1fx"
          % (D, len(E), p, r2, max(V) / min(V), max(col) / min(col)))
