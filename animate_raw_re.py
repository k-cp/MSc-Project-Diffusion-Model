"""Analyse + animate the RAW kf_vort_Re*_N256.npy files, directly on the cluster.

The files are (100 trajectories, 1000 frames, 256, 256) float64 -- ~52 GB each.
np.load(..., mmap_mode="r") reads ONLY the frames sliced out, so this touches a
few hundred MB per file, not 52 GB, and is safe on a login node.

For each file it prints an analysis block (shape, stats, frame-to-frame
decorrelation -- i.e. how finely time-sampled the file is) and writes one GIF of
a single trajectory, using matplotlib's pillow writer so no ffmpeg is needed.

    python animate_raw_re.py                       # all four Re, defaults
    RES=10000 TRAJ=5 STRIDE=8 python animate_raw_re.py
Env: RES (comma list, default all) / TRAJ / START / N_FRAMES / STRIDE / FPS
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

DATA_DIR = os.environ.get("DATA_DIR", "data")
RES = [int(r) for r in os.environ.get("RES", "500,1000,2000,10000").split(",")]
TRAJ = int(os.environ.get("TRAJ", 0))
START = int(os.environ.get("START", 0))
N_FRAMES = int(os.environ.get("N_FRAMES", 400))   # raw frames to span
STRIDE = int(os.environ.get("STRIDE", 5))         # raw files are finely sampled
FPS = int(os.environ.get("FPS", 12))


def main():
    for re_ in RES:
        path = os.path.join(DATA_DIR, f"kf_vort_Re{re_}_N256.npy")
        if not os.path.exists(path):
            print(f"{path}: not found, skipped")
            continue

        raw = np.load(path, mmap_mode="r")          # header only; nothing read yet
        print(f"\n=== {path}")
        print(f"  shape {raw.shape}  dtype {raw.dtype}  "
              f"({raw.size * raw.dtype.itemsize / 1e9:.1f} GB on disk)")

        hi = min(START + N_FRAMES, raw.shape[1])
        clip = np.asarray(raw[TRAJ, START:hi:STRIDE], dtype=np.float32)  # reads only this
        print(f"  trajectory {TRAJ}, frames {START}:{hi}:{STRIDE} -> {clip.shape[0]} frames")
        print(f"  std {clip.std():.4f}   min {clip.min():.2f}   max {clip.max():.2f}")

        # native time-sampling: rel-L2 between CONSECUTIVE RAW frames, plus across
        # the clip -- prints whether an animation of this slice will visibly move
        a, b = np.asarray(raw[TRAJ, START], dtype=np.float32), \
               np.asarray(raw[TRAJ, START + 1], dtype=np.float32)
        step = float(np.sqrt(((b - a) ** 2).mean()) / a.std())
        evo = float(np.sqrt(((clip[-1] - clip[0]) ** 2).mean()) / clip[0].std())
        print(f"  frame-to-frame change {step:.4f} rel-L2 of rms; "
              f"whole-clip evolution {evo:.3f} "
              f"({'visible' if evo > 0.25 else 'LOW -- raise N_FRAMES/STRIDE'})")

        vmax = float(np.percentile(np.abs(clip), 99.5))
        fig, ax = plt.subplots(figsize=(5.6, 5.9))
        im = ax.imshow(clip[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       origin="lower", animated=True)
        ax.set_xticks([]); ax.set_yticks([])
        title = ax.set_title(f"Re={re_}  traj {TRAJ}  frame {START}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("vorticity")

        def update(k, clip=clip, im=im, title=title, re_=re_):
            im.set_data(clip[k])
            title.set_text(f"Re={re_}  traj {TRAJ}  frame {START + k * STRIDE}")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=clip.shape[0],
                                      interval=1000 // FPS, blit=False)
        out = f"raw_re{re_}_traj{TRAJ}.gif"
        fig.tight_layout()
        ani.save(out, writer="pillow", fps=FPS, dpi=100)
        plt.close(fig)
        print(f"  saved {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
