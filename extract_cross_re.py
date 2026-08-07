"""Extract a small, pipeline-compatible TEST SET from the big cross-Reynolds files.

The kf_vort_Re*_N256.npy files are 52 GB each, float64, (100, 1000, 256, 256).
Nothing here needs that: a zero-shot evaluation only needs a test split. This
pulls a subset, converts to float32, and writes it in the layout the repo's
loaders expect, plus the matching sparse-sensor measurement file.

    python3 extract_cross_re.py kf_vort_Re500_N256.npy kf_vort_Re2000_N256.npy
    python3 extract_cross_re.py kf_vort_Re1000_N256.npy --traj 4 --frames 320

<<<<<<< HEAD
Defaults give 4 trajectories x 320 frames = 1272 sliding windows -- exactly the
size of the existing test set, so the numbers are directly comparable.
=======
Defaults give 8 trajectories x 320 frames. The LAST 4 are the test split (1272
sliding windows -- exactly the size of the existing test set, so the numbers are
directly comparable); the first 4 exist only so the loader can compute its
scaler from [:-4], which would otherwise be an empty slice and yield NaN.
>>>>>>> e4e07a3d5c8c3d39be0b13323776db25278eda9b

IMPORTANT -- these files are NOT the same simulations as the project's own
kf_2d_re1000_256_40seed.npy (same nominal Re=1000, but std 4.16 vs 4.85 and a
different forcing amplitude). So absolute scores on them will be worse than the
published ones for reasons that have nothing to do with Reynolds number. Use
the NEW Re1000 file as the reference point and read only the TREND across Re;
that holds the generator fixed and isolates the thing you actually want.

Each dataset is written with its OWN mean/std in a companion stats file. Reusing
km256_stats.npz would feed the model a ~14% scale shift and you would be
measuring that instead of the physics.
"""

import argparse
import os
import re

import numpy as np


def sensor_measurement(gt, n_sensors, seed):
    """Sparse-sensor low-res field: n random points, nearest-neighbour filled.

    Replicates DegradationSampler._sensor exactly (same Euclidean EDT fill), so
    the input distribution matches what the models were trained on. One sensor
    layout per trajectory, mirroring idx_lst in the original npz.
    """
    from scipy.ndimage import distance_transform_edt

    n_traj, n_frames, h, w = gt.shape
    out = np.empty_like(gt)
    idx_lst = np.empty((n_traj, n_sensors), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for t in range(n_traj):
        flat = rng.choice(h * w, size=n_sensors, replace=False)
        idx_lst[t] = flat
        mask = np.zeros(h * w, dtype=bool)
        mask[flat] = True
        iy, ix = distance_transform_edt(~mask.reshape(h, w),
                                        return_distances=False, return_indices=True)
        out[t] = gt[t][:, iy, ix]
        print(f"    trajectory {t}: {n_sensors} sensors placed")
    return out, idx_lst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
<<<<<<< HEAD
    p.add_argument("--traj", type=int, default=4, help="trajectories to keep (default 4)")
=======
    p.add_argument("--traj", type=int, default=8,
                   help="trajectories to keep (default 8). MUST be > 4: the repo's "
                        "loader takes [-4:] as the test split and computes its scaler "
                        "from [:-4], so a 4-trajectory file makes that slice EMPTY and "
                        "the scaler comes out NaN. With 8, the first 4 supply the "
                        "statistics (as the training split does normally) and the last "
                        "4 are the test set.")
>>>>>>> e4e07a3d5c8c3d39be0b13323776db25278eda9b
    p.add_argument("--frames", type=int, default=320, help="frames to keep (default 320)")
    p.add_argument("--start", type=int, default=0, help="first frame (skip transient if needed)")
    p.add_argument("--sensors", type=int, default=1024, help="sensor count (u3232 uses 1024)")
    p.add_argument("--outdir", default="data/cross_re")
    p.add_argument("--no-sensors", action="store_true",
                   help="ground truth only -- enough for SI, which manufactures its "
                        "own input via --si_eval_degradation")
    args = p.parse_args()

<<<<<<< HEAD
=======
    if args.traj <= 4:
        raise SystemExit(
            f"--traj {args.traj} is too few. load_recons_data takes [-4:] as the test\n"
            "split and derives its scaler from [:-4]; with 4 or fewer trajectories that\n"
            "slice is empty and mean/std come out NaN, so every output would be NaN.\n"
            "Use --traj 8 (4 for statistics + 4 for testing).")

>>>>>>> e4e07a3d5c8c3d39be0b13323776db25278eda9b
    os.makedirs(args.outdir, exist_ok=True)
    for path in args.files:
        if not os.path.exists(path):
            print(f"\n{path}: NOT FOUND"); continue
        tag = (re.search(r"Re(\d+)", path) or [None, "unknown"])[1]
        print(f"\n=== Re={tag}  ({path})")

        a = np.load(path, mmap_mode="r")           # header only
        sl = a[: args.traj, args.start : args.start + args.frames]
        gt = np.asarray(sl, dtype=np.float32)      # only this slice is read
        print(f"  extracted {gt.shape}  float64 -> float32  ({gt.nbytes/1e6:.0f} MB)")

        mean, std = float(gt.mean()), float(gt.std())
        print(f"  own statistics: mean {mean:+.5f}  std {std:.5f}")

        gt_path = os.path.join(args.outdir, f"kf_re{tag}_test.npy")
        np.save(gt_path, gt)
        stats_path = os.path.join(args.outdir, f"km256_stats_re{tag}.npz")
        np.savez(stats_path, mean=np.array(mean), scale=np.array(std))
        print(f"  wrote {gt_path}")
        print(f"  wrote {stats_path}   <- point config.data.stat_path here")

        if not args.no_sensors:
            print("  building sparse-sensor measurements (needs scipy)...")
            u, idx = sensor_measurement(gt, args.sensors, seed=abs(hash(tag)) % (2**31))
            npz_path = os.path.join(args.outdir, f"kf_re{tag}_sampled.npz")
            np.savez(npz_path, u3232=u, idx_lst=idx)
            # sanity check: the measurement MUST equal the truth at the sensor
            # points -- that is the defining property of this input (u3232 does).
            # Each trajectory has its OWN layout, so check per trajectory.
            uf = u.reshape(len(gt), gt.shape[1], -1)
            gf = gt.reshape(len(gt), gt.shape[1], -1)
            err = max(float(np.abs(uf[t][:, idx[t]] - gf[t][:, idx[t]]).max())
                      for t in range(len(gt)))
            flag = "OK" if err == 0.0 else "!! SHOULD BE 0 -- fill is wrong"
            print(f"  wrote {npz_path}  (exact at sensors: max err {err:.2e}  {flag})")

    print("\nNext: point a config at these files, e.g.")
    print("  data_dir:        ./data/cross_re/kf_re2000_test.npy")
    print("  sample_data_dir: ./data/cross_re/kf_re2000_sampled.npz")
    print("  stat_path:       ./data/cross_re/km256_stats_re2000.npz")
<<<<<<< HEAD
    print("\nNOTE the loaders take the LAST 4 trajectories as the test split, so with")
    print("--traj 4 the whole extracted file IS the test set (nothing is trained on).")
=======
    print("\nNOTE the loaders take the LAST 4 trajectories as the test split and compute")
    print("the scaler from the rest -- which is why 8 are extracted, not 4. Nothing is")
    print("trained on any of it; the first 4 only supply mean/std.")
>>>>>>> e4e07a3d5c8c3d39be0b13323776db25278eda9b


if __name__ == "__main__":
    main()
