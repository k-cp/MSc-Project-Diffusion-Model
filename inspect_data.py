"""Inspect .npy flow datasets WITHOUT loading them into memory.

Multi-GB arrays are opened with mmap_mode='r', so shape/dtype come from the
header alone and statistics are computed from a small slice. Nothing large is
ever read.

Purpose: before trusting a new Reynolds-number dataset for cross-Re work, check
it is structurally the SAME KIND of data as the one every current result rests
on -- same layout, same normalisation convention, same forcing, same time step.
A silently different convention would invalidate anything built on it.

    python inspect_data.py FILE [FILE ...]
    python inspect_data.py data/*.npy

Add --ref PATH to compare everything against a reference file (default: the
project's own ground truth, if present).
"""

import argparse
import os
import sys

import numpy as np

REF_DEFAULT = "data/kf_2d_re1000_256_40seed.npy"


def forcing_axis_report(w):
    """The Kolmogorov forcing is f = -4cos(4y): a k=4 band along ONE axis.

    Which axis carries it decides whether x-translation augmentation and the
    residual's meshgrid convention are still correct for this dataset.
    """
    out = {}
    for name, prof in (("rows(-2)", w.mean(axis=-1)), ("cols(-1)", w.mean(axis=-2))):
        amp = np.abs(np.fft.rfft(prof)) / len(prof) * 2
        out[name] = (int(amp[1:].argmax() + 1), float(amp[4]) if len(amp) > 4 else 0.0)
    return out


def inspect(path, ref_stats=None):
    if not os.path.exists(path):
        print(f"\n{path}\n  NOT FOUND")
        return None
    size_gb = os.path.getsize(path) / 1e9
    a = np.load(path, mmap_mode="r")          # header only -- nothing is read yet
    print(f"\n{os.path.basename(path)}   ({size_gb:.2f} GB on disk)")
    print(f"  shape {a.shape}   dtype {a.dtype}")

    if a.ndim != 4:
        print("  !! expected 4 dims (trajectories, frames, H, W) -- layout differs")
        return None
    n_traj, n_frames, h, w = a.shape
    print(f"  -> {n_traj} trajectories x {n_frames} frames of {h}x{w}")

    # statistics from a SMALL slice, not the whole array
    sample = np.asarray(a[0, : min(50, n_frames)], dtype=np.float64)
    mean, std = float(sample.mean()), float(sample.std())
    print(f"  values: mean {mean:+.4f}  std {std:.4f}  "
          f"min {float(sample.min()):+.2f}  max {float(sample.max()):+.2f}")

    tmean = sample.mean(axis=0)
    fa = forcing_axis_report(tmean)
    for name, (kdom, amp4) in fa.items():
        print(f"  mean-field along {name}: dominant k={kdom}, amplitude@k4={amp4:.3f}")
    carrier = max(fa, key=lambda k: fa[k][1])
    print(f"  -> forcing appears to lie along {carrier}"
          f"{'  (matches the reference)' if ref_stats and carrier == ref_stats['carrier'] else ''}")

    stats = {"shape": a.shape, "dtype": str(a.dtype), "mean": mean, "std": std,
             "carrier": carrier}
    if ref_stats:
        same_layout = a.shape[1:] == ref_stats["shape"][1:]
        print(f"  vs reference: frame layout {'SAME' if same_layout else 'DIFFERENT'}"
              f" | std ratio {std/ref_stats['std']:.2f}"
              f" | forcing axis {'same' if carrier == ref_stats['carrier'] else 'DIFFERENT'}")
        if not same_layout:
            print("     !! different frame shape -- the pipeline assumes (.., 256, 256)")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--ref", default=REF_DEFAULT,
                   help="reference dataset to compare against")
    args = p.parse_args()

    ref_stats = None
    if os.path.exists(args.ref):
        print("=" * 70)
        print(f"REFERENCE (what every current result is built on): {args.ref}")
        print("=" * 70)
        ref_stats = inspect(args.ref)
    else:
        print(f"(reference {args.ref} not found -- reporting absolute values only)")

    print("\n" + "=" * 70)
    print("CANDIDATE DATASETS")
    print("=" * 70)
    for f in args.files:
        if os.path.abspath(f) != os.path.abspath(args.ref):
            inspect(f, ref_stats)

    print("\nWhat to check before using any of these for cross-Re work:")
    print("  1. frame shape must be 256x256 (the model is fully convolutional but")
    print("     the residual's wavenumber grid and the stats file assume it)")
    print("  2. forcing must lie on the SAME axis, or x-translation augmentation")
    print("     and the residual's meshgrid convention are wrong for this data")
    print("  3. std should be broadly comparable -- a very different scale means a")
    print("     different normalisation, and km256_stats.npz would not transfer")
    print("  4. trajectories/frames counts set how the [:-4]/[-4:] split behaves")


if __name__ == "__main__":
    main()
