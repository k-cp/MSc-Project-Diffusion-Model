"""Animate the inpainted region over time: what each method puts in the hole.

Zooms on the dead square (plus a margin of context) so the reconstruction inside
it is actually visible — at full field scale a 64x64 hole is 6% of the frame and
the interesting behaviour is invisible.

The hole is fixed per trajectory, so the flow moves THROUGH a stationary window
of missing measurement. That is the point: a method that merely smooths will show
a static blur while the surroundings evolve; one that genuinely reconstructs will
show structure entering and leaving the hole in step with the flow.

    python animate_inpainting.py                    # empty (true inpainting)
    FILL=blk64 python animate_inpainting.py         # extrapolated variant
Env: FILL (blk64empty|blk64) TRAJECTORY FRAMES STRIDE FPS PAD OUT
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from functions.dead_block import apply_dead_block, block_for_trajectory

E = "experiments/kmflow_re1000_rs256_ddim_conditional_new"
REF = f"{E}/guided_recons_u3232_t400_r20_w0.0"
FILL = os.environ.get("FILL", "blk64empty")
TRAJ = int(os.environ.get("TRAJECTORY", 0))
FRAMES = int(os.environ.get("FRAMES", 318))
STRIDE = int(os.environ.get("STRIDE", 3))
FPS = int(os.environ.get("FPS", 12))
PAD = int(os.environ.get("PAD", 22))
SIZE, SEED, PER_TRAJ = 64, 0, 318
OUT = os.environ.get("OUT", f"inpainting_{FILL}.mp4")

METHODS = [("Diffusion baseline", "guided_recons_u3232_t400_r20_w0.0"),
           ("DPS ($\\zeta$=3.0)", "dps_guided_recons_u3232_t400_r20_w0.0_z3.0"),
           ("Stochastic interp.", "si_guided_recons_u3232_t400_r20_w0.0")]


def load(folder, fn, lo, hi):
    fs = sorted(glob.glob(os.path.join(folder, "sample_batch*", fn)),
                key=lambda p: int(os.path.basename(os.path.dirname(p))
                                  .replace("sample_batch", "")))
    if not fs:
        return None
    out, seen = [], 0
    for f in fs:
        a = np.load(f)
        n = a.shape[0]
        if seen + n > lo and seen < hi:
            out.append(a[max(0, lo - seen):min(n, hi - seen)])
        seen += n
        if seen >= hi:
            break
    return np.concatenate(out).astype(np.float32) if out else None


def main():
    lo, hi = TRAJ * PER_TRAJ, min(TRAJ * PER_TRAJ + FRAMES, (TRAJ + 1) * PER_TRAJ)
    y0, x0 = block_for_trajectory(SIZE, SEED, TRAJ, 256, 256)
    sl = (slice(max(0, y0 - PAD), min(256, y0 + SIZE + PAD)),
          slice(max(0, x0 - PAD), min(256, x0 + SIZE + PAD)))
    oy, ox = y0 - sl[0].start, x0 - sl[1].start
    print(f"{FILL}: trajectory {TRAJ}, frames {lo}-{hi} stride {STRIDE}, "
          f"hole at (y={y0}, x={x0})")

    ref = load(REF, "reference_arr.npy", lo, hi)[::STRIDE]
    inp_full = load(REF, "input_arr.npy", lo, hi)
    idx = np.load("idx_lst_test.npy")[TRAJ:TRAJ + 1]
    holed, _ = apply_dead_block(inp_full, idx, SIZE, SEED, inp_full.shape[0], 256, 256,
                                fill="zero" if FILL.endswith("empty") else "extrapolate")
    holed = holed[::STRIDE]

    panels = [("Ground truth", ref),
              ("Input (hole " + ("empty)" if FILL.endswith("empty") else "extrapolated)"), holed)]
    for label, sub in METHODS:
        a = load(f"{E}/{sub}_{FILL}", "sample_arr_run_0_it0.npy", lo, hi)
        if a is None:
            print(f"  {label}: missing")
            continue
        panels.append((label, a[::STRIDE]))
        print(f"  {label}: loaded")

    n = len(panels)
    v = float(np.percentile(np.abs(ref), 99.5))
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.5))
    ims = []
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(arr[0][sl], cmap="RdBu_r", vmin=-v, vmax=v,
                       origin="lower", animated=True)
        ax.add_patch(Rectangle((ox, oy), SIZE, SIZE, fill=False, ec="#2c7a5c", lw=2.0))
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        ims.append((im, arr))

    sup = fig.suptitle("", fontsize=9.5, y=0.03, color="#5d6b7c")

    def update(k):
        for im, arr in ims:
            im.set_data(arr[k][sl])
        sup.set_text(f"frame {lo + k * STRIDE}  ·  the flow moves through a stationary "
                     f"{SIZE}x{SIZE} region with no measurement (green)")
        return [im for im, _ in ims] + [sup]

    ani = animation.FuncAnimation(fig, update, frames=len(ref),
                                  interval=1000 // FPS, blit=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    ani.save(OUT, writer="ffmpeg" if OUT.endswith(".mp4") else "pillow", fps=FPS, dpi=115)
    print(f"Saved: {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, {len(ref)} frames)")


if __name__ == "__main__":
    main()
