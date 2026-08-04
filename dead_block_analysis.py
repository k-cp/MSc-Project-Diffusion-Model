"""Score the dead-block reconstructions INSIDE the hole versus outside it.

The hole is 6.25% of the field, so a global average is nearly blind to it: a
method could fail completely inside and still move overall MSE by only a few
percent. Everything here is therefore reported separately for the unobserved
region and the observed remainder, and against two references:

  the measurement  - what the input itself achieves in the hole (extrapolated
                     from the surviving boundary sensors). A method that does not
                     beat this has added nothing.
  the same method
  without a block  - how much that method loses by having the hole at all.

    python dead_block_analysis.py
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from functions.dead_block import block_for_trajectory

E = "experiments/kmflow_re1000_rs256_ddim_conditional_new"
SIZE, SEED, POS = 64, 0, None
PER_TRAJ, N_TRAJ = 318, 4
REF = f"{E}/guided_recons_u3232_t400_r20_w0.0"

RUNS = [
    ("Diffusion baseline", "#2f6ea8", "guided_recons_u3232_t400_r20_w0.0"),
    ("DPS ($\\zeta$=3.0)", "#7a5296", "dps_guided_recons_u3232_t400_r20_w0.0_z3.0"),
    ("Stochastic interp.", "#b04a2f", "si_guided_recons_u3232_t400_r20_w0.0"),
]
OUT = "dead_block_results.png"


def load(folder, fn="sample_arr_run_0_it0.npy"):
    fs = sorted(glob.glob(os.path.join(folder, "sample_batch*", fn)),
                key=lambda p: int(os.path.basename(os.path.dirname(p))
                                  .replace("sample_batch", "")))
    return np.concatenate([np.load(f) for f in fs]).astype(np.float64) if fs else None


def hole_mask(n):
    """(N, h, w) boolean: True inside that sample's trajectory's dead square."""
    m = np.zeros((n, 256, 256), dtype=bool)
    for t in range(N_TRAJ):
        y0, x0 = block_for_trajectory(SIZE, SEED, t, 256, 256, POS)
        lo, hi = t * PER_TRAJ, min((t + 1) * PER_TRAJ, n)
        m[lo:hi, y0:y0 + SIZE, x0:x0 + SIZE] = True
    return m


def rms(a):
    return float(np.sqrt((a ** 2).mean()))


def main():
    ref = load(REF, "reference_arr.npy")
    inp = load(REF, "input_arr.npy")            # the ORIGINAL (unholed) measurement
    # The floor a method must beat is NOT the unholed input -- that still has
    # sensors in the region. It is the HOLED input: the boundary extrapolation a
    # method is handed. Rebuild it here from the same code the runs used.
    holed = None
    if inp is not None and os.path.exists("idx_lst_test.npy"):
        from functions.dead_block import apply_dead_block
        holed, _ = apply_dead_block(inp, np.load("idx_lst_test.npy"), SIZE, SEED,
                                    PER_TRAJ, 256, 256, pos=POS)
    n = ref.shape[0]
    hole = hole_mask(n)
    s = ref.std()
    print(f"{n} samples, hole = {100 * hole.mean():.2f}% of all pixels, field rms {s:.3f}\n")

    rows = []
    # the measurement's own error, with and without the hole, as the floor
    print(f"{'':34s} {'INSIDE hole':>12} {'outside':>10} {'global':>9}")
    print("-" * 68)
    if inp is not None:
        print(f"{'measurement, no block':34s} {rms((inp-ref)[hole])/s:12.3f} "
              f"{rms((inp-ref)[~hole])/s:10.3f} {rms(inp-ref)/s:9.3f}")
    if holed is not None:
        print(f"{'measurement WITH block (the floor)':34s} {rms((holed-ref)[hole])/s:12.3f} "
              f"{rms((holed-ref)[~hole])/s:10.3f} {rms(holed-ref)/s:9.3f}")

    for label, colour, sub in RUNS:
        base = load(f"{E}/{sub}")
        blk = load(f"{E}/{sub}_blk64")
        if blk is None:
            print(f"{label:34s} MISSING _blk64 run")
            continue
        r = {"label": label, "colour": colour,
             "in": rms((blk - ref)[hole]) / s, "out": rms((blk - ref)[~hole]) / s,
             "glob": rms(blk - ref) / s,
             "in0": rms((base - ref)[hole]) / s if base is not None else np.nan,
             "glob0": rms(base - ref) / s if base is not None else np.nan}
        rows.append(r)
        print(f"{label + ' + dead block':34s} {r['in']:12.3f} {r['out']:10.3f} {r['glob']:9.3f}")
        print(f"{'   (same method, no block)':34s} {r['in0']:12.3f} "
              f"{rms((base-ref)[~hole])/s if base is not None else np.nan:10.3f} {r['glob0']:9.3f}")

    print("\nWhat the hole costs each method, and whether it beats the raw input there:")
    print(f"{'':24s} {'in-hole err':>12} {'vs no block':>12} {'vs input':>10}")
    floor = rms((holed - ref)[hole]) / s if holed is not None else None
    for r in rows:
        vs_in = floor
        print(f"  {r['label']:22s} {r['in']:12.3f} {r['in']/r['in0']:11.2f}x "
              f"{r['in']/vs_in if vs_in else float('nan'):9.2f}x")
    print("\n  'vs input' < 1 means the method improved on the measurement inside the hole;")
    print("  > 1 means it did worse than simply extrapolating from the boundary.")

    # ---------------- figure ----------------
    if not rows:
        return
    fig = plt.figure(figsize=(13.6, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.85], hspace=0.3, wspace=0.1)

    t, k = 0, 40                                  # trajectory 0, some frame
    y0, x0 = block_for_trajectory(SIZE, SEED, t, 256, 256, POS)
    v = float(np.percentile(np.abs(ref[k]), 99.5))
    panels = [("Ground truth", ref[k])]
    for label, colour, sub in RUNS:
        a = load(f"{E}/{sub}_blk64")
        if a is not None:
            panels.append((label, a[k]))
    for j, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax.add_patch(Rectangle((x0, y0), SIZE, SIZE, fill=False, ec="#2c7a5c", lw=2.0))
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, :2])
    xs = np.arange(len(rows))
    ax.bar(xs - 0.19, [r["in0"] for r in rows], 0.36, label="no block",
           color=[r["colour"] for r in rows], alpha=0.45, edgecolor="white")
    ax.bar(xs + 0.19, [r["in"] for r in rows], 0.36, label="dead block",
           color=[r["colour"] for r in rows], edgecolor="white")
    if holed is not None:
        ax.axhline(rms((holed - ref)[hole]) / s, color="#9a7218", ls="--", lw=1.3)
        ax.text(len(rows) - 0.5, rms((holed - ref)[hole]) / s, " holed input (the floor)",
                color="#7a5a18", fontsize=9, va="bottom", ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.set_ylabel("normalised error INSIDE the hole")
    ax.set_title("Error in the unobserved region", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    ax = fig.add_subplot(gs[1, 2:])
    for r in rows:
        ax.plot([0, 1], [r["out"], r["in"]], marker="o", ms=8, lw=2,
                color=r["colour"], label=r["label"])
        ax.text(1.03, r["in"], f"{r['in']:.3f}", color=r["colour"], fontsize=9.5, va="center")
    ax.set_xlim(-0.15, 1.32); ax.set_xticks([0, 1])
    ax.set_xticklabels(["outside the hole", "inside the hole"], fontsize=9.5)
    ax.set_ylabel("normalised error")
    ax.set_title("What the hole costs", fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    fig.suptitle(f"Reconstruction with a {SIZE}x{SIZE} dead sensor block "
                 f"({100 * SIZE * SIZE / 65536:.2f}% of the field, no measurement inside)",
                 fontsize=13, x=0.008, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(OUT, dpi=170)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
