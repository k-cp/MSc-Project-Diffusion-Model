"""Animate ground truth against each method's reconstruction, side by side.

Unlike animate_results.py this needs no input_arr.npy (which is absent from most
run folders locally) -- it uses reference_arr and each method's sample only.

IMPORTANT -- the 1272 saved frames are FOUR test trajectories concatenated
(318 frames each), so animating straight through produces three hard cuts where
one simulation ends and the next begins. TRAJECTORY selects one so the motion is
continuous.

    python animate_comparison.py                 # 4-panel mp4, first trajectory
    FRAMES=120 python animate_comparison.py      # shorter
    DIFF=1 python animate_comparison.py          # add an error row
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Which experiment tree: the main benchmark, or any cross_re* directory.
#   EXP=cross_re10000 python animate_comparison.py
EXP = os.environ.get("EXP", "kmflow_re1000_rs256_ddim_conditional_new")
E = os.path.join("experiments", EXP)

# Every method that might be present. Missing folders are skipped with a note, so
# the same command works on the main tree (which has the retrained baselines) and
# on the cross_re trees (which do not).
CANDIDATES = [
    ("Baseline (provided)",  "guided_recons_u3232_t400_r20_w0.0"),
    ("Baseline (retrained)", "guided_recons_u3232_t400_r20_w0.0_mine"),
    ("Baseline (+ x-shift)", "guided_recons_u3232_t400_r20_w0.0_mine_xshift"),
    ("DPS ($\\zeta$=3.0)",   "dps_guided_recons_u3232_t400_r20_w0.0_z3.0"),
    ("Stochastic interp.",   "si_guided_recons_u3232_t400_r20_w0.0"),
]
METHODS = [(lab, os.path.join(E, sub)) for lab, sub in CANDIDATES
           if glob.glob(os.path.join(E, sub, "sample_batch0", "sample_arr_run_0_it0.npy"))]

# The reference is identical across methods within an experiment, but only some
# folders were transferred with it — take the first that has it.
REF_FROM = next((os.path.join(E, sub) for _, sub in CANDIDATES
                 if os.path.exists(os.path.join(E, sub, "sample_batch0", "reference_arr.npy"))),
                os.path.join(E, CANDIDATES[0][1]))

TRAJECTORY = int(os.environ.get("TRAJECTORY", 0))     # 0-3
FRAMES = int(os.environ.get("FRAMES", 318))           # data frames to SPAN, up to 318
FPS = int(os.environ.get("FPS", 20))

# STRIDE matters, because the datasets are NOT sampled at the same rate in time.
# Successive frames of the main benchmark differ by 0.042 of the field rms; for the
# cross_re* sets it is 0.0035 -- about 12x finer. Animating cross-Re data at
# stride 1 therefore looks frozen: 120 frames of it covers as much evolution as
# 10 frames of the main set. Default to spanning the whole trajectory instead.
STRIDE = int(os.environ.get("STRIDE", 1 if EXP.startswith("kmflow") else 3))
DIFF = bool(int(os.environ.get("DIFF", 0)))
OUT = os.environ.get("OUT", "comparison.mp4")
PER_TRAJ = 318


def load(folder, fn, lo, hi):
    """Frames [lo, hi) of a run, walking only the batches that overlap them."""
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", fn)),
                   key=lambda p: int(os.path.basename(os.path.dirname(p)).replace("sample_batch", "")))
    if not files:
        return None
    out, seen = [], 0
    for f in files:
        a = np.load(f)
        n = a.shape[0]
        if seen + n > lo and seen < hi:
            out.append(a[max(0, lo - seen):min(n, hi - seen)])
        seen += n
        if seen >= hi:
            break
    return np.concatenate(out).astype(np.float32) if out else None


def main():
    lo = TRAJECTORY * PER_TRAJ
    hi = min(lo + FRAMES, (TRAJECTORY + 1) * PER_TRAJ)
    n_anim = len(range(0, hi - lo, STRIDE))
    print(f"{EXP}: trajectory {TRAJECTORY}, data frames {lo}–{hi} "
          f"at stride {STRIDE} -> {n_anim} animation frames")

    ref = load(REF_FROM, "reference_arr.npy", lo, hi)
    if ref is None:
        print(f"No reference_arr under {REF_FROM}")
        return
    ref = ref[::STRIDE]
    evo = float(np.sqrt(((ref[-1] - ref[0]) ** 2).mean()) / ref[0].std())
    print(f"  evolution across the clip: {evo:.3f} rel-L2 of the field rms "
          f"({'visible' if evo > 0.25 else 'LOW — raise FRAMES or STRIDE'})")
    panels = [("Ground truth", ref)]
    for label, d in METHODS:
        a = load(d, "sample_arr_run_0_it0.npy", lo, hi)
        if a is None:
            print(f"  {label}: missing, skipped")
            continue
        panels.append((label, a[::STRIDE]))
        print(f"  {label}: loaded {a[::STRIDE].shape}")

    n = len(panels)
    rows = 2 if DIFF else 1
    fig, axes = plt.subplots(rows, n, figsize=(3.3 * n, 3.5 * rows), squeeze=False)

    # One symmetric colour scale for every panel, so differences in amplitude are
    # visible rather than normalised away per-panel.
    vmax = float(np.percentile(np.abs(ref), 99.5))
    ims = []
    for j, (label, a) in enumerate(panels):
        ax = axes[0][j]
        ims.append((ax.imshow(a[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              origin="lower", animated=True), a, False))
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    if DIFF:
        dmax = vmax * 0.6
        for j, (label, a) in enumerate(panels):
            ax = axes[1][j]
            if j == 0:
                ax.axis("off")
                ims.append(None)
                continue
            ims.append((ax.imshow(a[0] - ref[0], cmap="PuOr_r", vmin=-dmax, vmax=dmax,
                                  origin="lower", animated=True), a, True))
            ax.set_title(f"error — {label}", fontsize=9.5, color="#5d6b7c")
            ax.set_xticks([]); ax.set_yticks([])

    sup = fig.suptitle("", fontsize=10, y=0.02, color="#5d6b7c")

    def update(k):
        artists = []
        for entry in ims:
            if entry is None:
                continue
            im, a, is_diff = entry
            im.set_data(a[k] - ref[k] if is_diff else a[k])
            artists.append(im)
        sup.set_text(f"frame {lo + k * STRIDE}  ·  trajectory {TRAJECTORY}  ·  {EXP}  ·  1024 sensors")
        artists.append(sup)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(ref), interval=1000 // FPS,
                                  blit=False)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    writer = "ffmpeg" if OUT.endswith(".mp4") else "pillow"
    print(f"Rendering {len(ref)} frames -> {OUT} ...")
    ani.save(OUT, writer=writer, fps=FPS, dpi=110)
    print(f"Saved: {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
