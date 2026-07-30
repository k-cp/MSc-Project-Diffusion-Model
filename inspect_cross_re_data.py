"""Characterise the cross-Reynolds datasets: snapshots, spectra, distributions, stats.

These are the test splits extracted by extract_cross_re.py from the big
kf_vort_Re*_N256.npy files, and they are already on disk as the reference_arr
of each cross_re* run -- there is no need to touch the 52 GB originals to see
what they look like.

The point of the figure is the caveat that governs Section 5.8: this family is
NOT the generator the models trained on. Its Re=1000 case has a different
variance from the project's own Re=1000 data (4.38 vs 4.76), so absolute scores
on it are not comparable to the published ones and only the trend is meaningful.

    python inspect_cross_re_data.py
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_ke_spectrum

RES = [500, 1000, 2000, 10000]
BASE = "guided_recons_u3232_t400_r20_w0.0"
MAIN = ("experiments/kmflow_re1000_rs256_ddim_conditional_new/"
        "guided_recons_u3232_t400_r20_w0.0")
OUT = "cross_re_data.png"
FRAME = 0          # which field within the first batch to show


def load(folder):
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", "reference_arr.npy")))
    if not files:
        return None
    return np.concatenate([np.load(f) for f in files]).astype(np.float64)


def mean_spectrum(fields, stride):
    tot, k = None, None
    sub = fields[::stride]
    for f in sub:
        k, ek = compute_ke_spectrum(f)
        tot = ek if tot is None else tot + ek
    return k, tot / len(sub)


def main():
    data = {}
    for r in RES:
        d = os.path.join("experiments", f"cross_re{r}", BASE)
        a = load(d)
        if a is None:
            print(f"  Re{r}: no reference_arr, skipped")
            continue
        data[r] = a
        print(f"  Re{r:<6} shape={a.shape}  std={a.std():.4f}  "
              f"min={a.min():7.2f} max={a.max():7.2f}  "
              f"kurtosis={float(((a-a.mean())**4).mean()/a.var()**2 - 3):.3f}")

    main_ref = load(MAIN)
    if main_ref is not None:
        print(f"  {'TRAINING DATA (project Re=1000)':<28} std={main_ref.std():.4f}  "
              f"min={main_ref.min():7.2f} max={main_ref.max():7.2f}")

    if not data:
        print("Nothing to plot.")
        return

    n = len(data)
    fig = plt.figure(figsize=(4.0 * n, 8.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1.05, 0.82], hspace=0.28, wspace=0.12)

    # --- row 1: a snapshot per Reynolds number, on a SHARED colour scale so the
    # growth of small-scale structure is visible rather than normalised away ---
    vmax = max(abs(np.percentile(a[FRAME], [1, 99])).max() for a in data.values())
    for i, (r, a) in enumerate(sorted(data.items())):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(a[FRAME], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"Re = {r}", fontsize=12, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.055, f"std {a.std():.3f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.5, color="#5d6b7c")
        if i == n - 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label("vorticity $\\omega$", fontsize=9)

    colours = plt.cm.viridis(np.linspace(0.08, 0.85, n))

    # --- row 2a: energy spectra ---
    ax = fig.add_subplot(gs[1, :max(1, n // 2)])
    for c, (r, a) in zip(colours, sorted(data.items())):
        k, e = mean_spectrum(a, stride=max(1, len(a) // 64))
        ax.loglog(k, e, lw=1.7, color=c, label=f"Re = {r}")
    if main_ref is not None:
        k, e = mean_spectrum(main_ref, stride=max(1, len(main_ref) // 64))
        ax.loglog(k, e, lw=1.7, color="#b04a2f", ls="--",
                  label="training data (Re = 1000)")
    ax.set_xlabel("wavenumber $k$"); ax.set_ylabel("$E(k)$")
    ax.set_title("Energy spectra", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.22, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- row 2b: vorticity distributions ---
    ax = fig.add_subplot(gs[1, max(1, n // 2):])
    bins = np.linspace(-25, 25, 240)
    for c, (r, a) in zip(colours, sorted(data.items())):
        h, _ = np.histogram(a[::7], bins=bins, density=True)
        ax.semilogy(0.5 * (bins[1:] + bins[:-1]), h, lw=1.6, color=c, label=f"Re = {r}")
    if main_ref is not None:
        h, _ = np.histogram(main_ref[::7], bins=bins, density=True)
        ax.semilogy(0.5 * (bins[1:] + bins[:-1]), h, lw=1.6, color="#b04a2f", ls="--",
                    label="training data")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("vorticity $\\omega$"); ax.set_ylabel("$p(\\omega)$")
    ax.set_title("Vorticity distributions", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("The cross-Reynolds evaluation datasets", fontsize=14,
                 x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.012,
             "Test splits extracted from kf_vort_Re*_N256. The dashed red curve is the "
             "project's OWN Re=1000 training data — a different generator (std 4.76 vs 4.38 "
             "at the same nominal Re), which is why only the trend across Re is meaningful.",
             fontsize=8.5, color="#5d6b7c")
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
