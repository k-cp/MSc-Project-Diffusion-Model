"""Validate the dead-block measurement against the shipped data. Run on the cluster.

Two checks, in order of importance:

  1. With size=0 the construction must reproduce u3232 BIT FOR BIT. u3232 is
     itself the nearest-neighbour fill of the same 1024 sensors, so if our fill
     disagrees with it the operator is wrong and every dead-block number would
     be meaningless. This is the check that licenses the experiment.

  2. With a real block, report how much of the array dies and confirm no
     surviving sensor lies inside the hole.

    python verify_dead_block.py                 # default config, block 64
    BLOCK=96 python verify_dead_block.py
"""

import argparse
import os

import numpy as np
import yaml

from functions.dead_block import (apply_dead_block, block_for_trajectory,
                                  surviving_sensors)

CONFIG = os.environ.get("CONFIG", "kmflow_re1000_rs256_conditional.yml")
SIZE = int(os.environ.get("BLOCK", 64))
SEED = int(os.environ.get("BLOCK_SEED", 0))
N_CHECK = int(os.environ.get("N_CHECK", 20))     # frames per trajectory for check 1


def main():
    with open(os.path.join("configs", CONFIG)) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    print(f"config {CONFIG}\n  gt  : {d['data_dir']}\n  meas: {d['sample_data_dir']}")

    gt_all = np.load(d["data_dir"], mmap_mode="r")
    with np.load(d["sample_data_dir"], allow_pickle=True) as f:
        idx = f["idx_lst"][-4:].astype(np.int64)
        u = f[d["data_kw"]][-4:, :N_CHECK]                 # (4, N_CHECK, 256, 256)
    gt = np.asarray(gt_all[-4:, :N_CHECK], dtype=np.float32)
    n_traj, per, h, w = gt.shape
    print(f"  test trajectories {n_traj}, sensors/traj {idx.shape[1]}, grid {h}x{w}")

    # ---- check 1: size=0 must reproduce u3232 -------------------------------
    flat_gt = gt.reshape(n_traj * per, h, w)
    blur, _ = apply_dead_block(flat_gt, idx, 0, 0, per, h, w)
    diff = np.abs(blur - u.reshape(n_traj * per, h, w).astype(np.float32))
    print(f"\n[1] size=0 vs shipped u3232: max|diff| = {diff.max():.3e}, "
          f"mean|diff| = {diff.mean():.3e}")
    if diff.max() < 1e-5:
        print("    PASS - the fill reproduces the shipped measurement.")
    else:
        print("    *** FAIL - the operator does NOT match u3232. Do not trust any")
        print("    *** dead-block result until this is resolved.")
        raise SystemExit(1)

    # ---- check 2: a real block ---------------------------------------------
    print(f"\n[2] block {SIZE}x{SIZE}, seed {SEED}:")
    tot_dead = 0
    for t in range(n_traj):
        b = block_for_trajectory(SIZE, SEED, t, h, w)
        keep, dead = surviving_sensors(idx[t], b, SIZE, h, w)
        iy, ix = np.divmod(keep, w)
        inside = ((iy >= b[0]) & (iy < b[0] + SIZE) &
                  (ix >= b[1]) & (ix < b[1] + SIZE)).sum()
        tot_dead += dead
        print(f"    traj {t}: block at (y={b[0]:3d}, x={b[1]:3d})  "
              f"dead {dead:4d}  surviving {len(keep):4d}  "
              f"survivors inside hole {inside} (must be 0)")
        assert inside == 0
    frac_area = 100.0 * SIZE * SIZE / (h * w)
    print(f"    hole covers {frac_area:.2f}% of the field; "
          f"{tot_dead / n_traj:.0f} sensors lost on average "
          f"({100.0 * tot_dead / n_traj / idx.shape[1]:.1f}% of the array)")

    blur_b, surv = apply_dead_block(flat_gt, idx, SIZE, SEED, per, h, w)
    b0 = block_for_trajectory(SIZE, SEED, 0, h, w)
    hole = blur_b[:per, b0[0]:b0[0] + SIZE, b0[1]:b0[1] + SIZE]
    true = flat_gt[:per, b0[0]:b0[0] + SIZE, b0[1]:b0[1] + SIZE]
    err = np.sqrt(((hole - true) ** 2).mean()) / flat_gt.std()
    print(f"\n    normalised error of the FILLED hole vs truth (traj 0): {err:.3f}")
    print("    (this is what a method must beat; the fill just extrapolates the boundary)")


if __name__ == "__main__":
    main()
