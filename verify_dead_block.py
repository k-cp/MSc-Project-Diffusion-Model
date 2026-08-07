"""Validate the dead-block measurement against the real data. Run on the cluster.

The dead block patches the shipped u3232 in place, so the invariants to check are
about what it does and does NOT touch:

  1. size=0 leaves the measurement bit-identical (no accidental perturbation).
  2. With a real block, everything OUTSIDE the hole is bit-identical, so these
     runs remain comparable with every other result in the project.
  3. The hole actually changes, and no surviving sensor lies inside it.
  4. Report the naive filled hole's error — the bar a method must beat.

An earlier version rebuilt the whole field from surviving sensors; this gate
caught that it disagreed with u3232 on 4.8% of pixels (tie-breaking in the
nearest-neighbour fill). Do not remove these checks.

    python verify_dead_block.py                       # block 64, random position
    BLOCK=96 BLOCK_POS=center python verify_dead_block.py
"""

import os

import numpy as np
import yaml

from functions.dead_block import (apply_dead_block, block_for_trajectory, parse_size,
                                  read_npz_trajectories, surviving_sensors)

CONFIG = os.environ.get("CONFIG", "kmflow_re1000_rs256_conditional.yml")
SIZE = os.environ.get("BLOCK", "64")   # int, "HxW" or "P%" - see dead_block.parse_size
SEED = int(os.environ.get("BLOCK_SEED", 0))
POS = os.environ.get("BLOCK_POS", "") or None
N_CHECK = int(os.environ.get("N_CHECK", 20))     # frames per trajectory


def main():
    with open(os.path.join("configs", CONFIG)) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    print(f"config {CONFIG}\n  meas: {d['sample_data_dir']}\n  gt  : {d['data_dir']}")

    # NpzFile[key] would decompress all ~3.4 GB before slicing; read just the
    # trajectories we need (this OOM-killed a login-node process once already).
    with np.load(d["sample_data_dir"], allow_pickle=True) as f:
        idx = f["idx_lst"][-4:].astype(np.int64)
    u = read_npz_trajectories(d["sample_data_dir"], d["data_kw"],
                              slice(-4, None), n_frames=N_CHECK)
    gt = np.asarray(np.load(d["data_dir"], mmap_mode="r")[-4:, :N_CHECK],
                    dtype=np.float32)
    n_traj, per, h, w = u.shape
    flat_u = u.reshape(n_traj * per, h, w).astype(np.float32)
    flat_gt = gt.reshape(n_traj * per, h, w)
    print(f"  {n_traj} test trajectories, {idx.shape[1]} sensors each, grid {h}x{w}")

    # sanity: the measurement must be exact at the sensors, or nothing else holds
    iy, ix = np.divmod(idx[0], w)
    at_sensors = np.abs(flat_u[:per][:, iy, ix] - flat_gt[:per][:, iy, ix]).max()
    print(f"\n[0] measurement equals ground truth at the sensors: "
          f"max|diff| = {at_sensors:.3e}")
    assert at_sensors == 0.0, "u3232 is not exact at idx_lst - check the index convention"

    # ---- 1. size=0 must be a no-op -----------------------------------------
    zero, _ = apply_dead_block(flat_u, idx, 0, 0, per, h, w)
    d0 = float(np.abs(zero - flat_u).max())
    print(f"[1] size=0 leaves the measurement untouched: max|diff| = {d0:.3e}  "
          f"{'PASS' if d0 == 0.0 else '*** FAIL'}")
    if d0 != 0.0:
        raise SystemExit(1)

    # ---- 2/3. a real block --------------------------------------------------
    BH, BW = parse_size(SIZE, h, w)
    blur, surv = apply_dead_block(flat_u, idx, SIZE, SEED, per, h, w, pos=POS)
    print(f"\n[2] block {BH}x{BW}, position={POS or f'random (seed {SEED})'}:")
    ok = True
    for t in range(n_traj):
        b = block_for_trajectory(SIZE, SEED, t, h, w, POS)
        y0, x0 = b
        keep, dead = surviving_sensors(idx[t], b, SIZE, h, w)
        ky, kx = np.divmod(keep, w)
        inside = int(((ky >= y0) & (ky < y0 + BH) &
                      (kx >= x0) & (kx < x0 + BW)).sum())
        sl = slice(t * per, (t + 1) * per)
        out = np.ones((h, w), bool); out[y0:y0 + BH, x0:x0 + BW] = False
        d_out = float(np.abs(blur[sl][:, out] - flat_u[sl][:, out]).max())
        d_in = float(np.abs(blur[sl][:, y0:y0 + BH, x0:x0 + BW]
                            - flat_u[sl][:, y0:y0 + BH, x0:x0 + BW]).mean())
        good = (d_out == 0.0) and (inside == 0) and (d_in > 0)
        ok &= good
        print(f"    traj {t}: block (y={y0:3d}, x={x0:3d})  dead {dead:4d}  "
              f"surviving {len(keep):4d}  in-hole survivors {inside}  "
              f"outside max|diff| {d_out:.1e}  inside mean|diff| {d_in:.3f}  "
              f"{'PASS' if good else '*** FAIL'}")
    if not ok:
        print("\n*** FAIL - do not trust any dead-block result until resolved.")
        raise SystemExit(1)

    # ---- 4. the bar to beat -------------------------------------------------
    b = block_for_trajectory(SIZE, SEED, 0, h, w, POS)
    y0, x0 = b
    sl = slice(0, per)
    hole = blur[sl][:, y0:y0 + BH, x0:x0 + BW]
    true = flat_gt[sl][:, y0:y0 + BH, x0:x0 + BW]
    e_in = float(np.sqrt(((hole - true) ** 2).mean()) / flat_gt.std())
    out = np.ones((h, w), bool); out[y0:y0 + BH, x0:x0 + BW] = False
    e_out = float(np.sqrt(((blur[sl][:, out] - flat_gt[sl][:, out]) ** 2).mean())
                  / flat_gt.std())
    print(f"\n[4] input error (normalised by field rms), trajectory 0:")
    print(f"    inside the hole {e_in:.3f}   outside {e_out:.3f}   "
          f"-> the hole is {e_in / e_out:.1f}x worse")
    print(f"    hole covers {100.0 * BH * BW / (h * w):.2f}% of the field")
    print("\nALL CHECKS PASSED - the dead-block measurement is sound.")


if __name__ == "__main__":
    main()
