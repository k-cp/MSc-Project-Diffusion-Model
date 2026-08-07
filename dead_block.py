"""Dead-sensor-block measurement: a square patch of the sensor array fails.

Motivation. Every other experiment in this project degrades the measurement
*uniformly* — fewer sensors, coarser sampling, a low-pass filter. In all of them
each point of the field is near some constraint. A dead block is different: it
creates a region with NO measurement information at all, so the reconstruction
inside it must come entirely from the learned prior and from the flow's
dynamics, not from interpolation.

That distinction matters here. SI is pinned by exact sensor values and, on this
benchmark, behaves close to a conditional-mean estimator; the diffusion methods
noise the measurement away and generate. Inside the hole the first has nothing
to be pinned by, so this is the configuration where the ranking could plausibly
invert — which is the point of running it.

CONSISTENCY REQUIREMENT. Two things must be masked together, or the experiment
is void:
  1. the input field  — rebuilt by nearest-neighbour fill from the SURVIVING
     sensors only, so the hole is extrapolated from its boundary (exactly what a
     real dead patch produces);
  2. DPS's sensor index set — DPS gathers ground truth at idx_lst directly
     (posterior_sampling.py), so leaving those indices in place would let it read
     the true field inside the hole.

SELF-CHECK. With size=0 this reproduces the stock u3232 measurement bit for bit
(u3232 is itself the nearest-neighbour fill of the same 1024 sensors), so the
machinery can be validated against the shipped data before trusting any result.
"""

import numpy as np


def block_for_trajectory(size, seed, traj, h=256, w=256, pos=None):
    """Top-left corner of the dead square for one trajectory.

    One block per TRAJECTORY, not per frame: a sensor patch fails and stays
    failed, so the hole must sit still while the flow evolves through it.

    pos controls WHERE:
      None / ""      random per trajectory, seeded on (seed, traj) — different
                     independent holes, which averages over flow features and is
                     the right default for a quantitative result.
      "center"       centred in the field, identical for every trajectory.
      "y,x"          explicit top-left corner, identical for every trajectory —
                     use this when a figure must show the same region every time.
    """
    if size <= 0:
        return None
    if pos:
        if str(pos).strip() == "center":
            return (h - size) // 2, (w - size) // 2
        y, _, x = str(pos).partition(",")
        y, x = int(y), int(x)
        if not (0 <= y <= h - size and 0 <= x <= w - size):
            raise ValueError(f"--block_pos {pos!r} puts a {size}x{size} block outside "
                             f"the {h}x{w} field")
        return y, x
    rng = np.random.default_rng(int(seed) * 100003 + int(traj))
    return int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))


def surviving_sensors(flat_idx, block, size, h=256, w=256):
    """Drop the sensors that fall inside the dead square.

    flat_idx : (n,) flat indices into the h*w grid (one trajectory's layout).
    Returns the surviving subset, and how many died.
    """
    if block is None or size <= 0:
        return flat_idx, 0
    y0, x0 = block
    iy, ix = np.divmod(np.asarray(flat_idx), w)
    dead = (iy >= y0) & (iy < y0 + size) & (ix >= x0) & (ix < x0 + size)
    return np.asarray(flat_idx)[~dead], int(dead.sum())


def voronoi_fill(field, flat_idx, h=256, w=256):
    """Nearest-neighbour fill of `field` from the given sensor locations.

    field : (..., h, w) ground truth. Returns an array of the same shape whose
    value at every pixel is the field's value at the nearest sensor — i.e. the
    Voronoi/nearest-neighbour reconstruction that u3232 already is.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.zeros(h * w, dtype=bool)
    mask[np.asarray(flat_idx)] = True
    mask = mask.reshape(h, w)
    if not mask.any():
        raise ValueError("no surviving sensors — the dead block covers them all")
    # indices of the nearest measured pixel for every location
    iy, ix = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return field[..., iy, ix]


def apply_dead_block(gt, idx_per_traj, size, seed, per_traj, h=256, w=256,
                     log=None, pos=None):
    """Build the dead-block measurement for a whole flattened test set.

    gt           : (N, C, h, w) ground truth, N = n_traj * per_traj samples
                   laid out trajectory-major (the repo's flattening order).
    idx_per_traj : (n_traj, n_sensors) sensor layout for each trajectory.
    per_traj     : samples per trajectory (318 for this benchmark).

    Returns (blur, surviving_flat) where blur has gt's shape and
    surviving_flat is a list, one entry per trajectory, of the sensors that
    remain — DPS must be given these, not the originals.
    """
    gt = np.asarray(gt)
    n_traj = len(idx_per_traj)
    if n_traj * per_traj != gt.shape[0]:
        raise ValueError(f"{n_traj} trajectories x {per_traj} != {gt.shape[0]} samples")

    blur = np.empty_like(gt)
    surviving = []
    for t in range(n_traj):
        block = block_for_trajectory(size, seed, t, h, w, pos)
        keep, n_dead = surviving_sensors(idx_per_traj[t], block, size, h, w)
        surviving.append(keep)
        lo, hi = t * per_traj, (t + 1) * per_traj
        blur[lo:hi] = voronoi_fill(gt[lo:hi], keep, h, w)
        if log is not None:
            where = "none" if block is None else f"at (y={block[0]}, x={block[1]})"
            log(f"  trajectory {t}: block {size}x{size} {where}; "
                f"{n_dead}/{len(idx_per_traj[t])} sensors dead, {len(keep)} surviving")
    return blur, surviving


def build_from_config(args, config, ref_data, log=None, idx_kw="idx_lst"):
    """Whole dead-block measurement for a runner, from the flattened test set.

    ref_data : (N, C, h, w) torch tensor or array — the FLATTENED test split, laid
    out trajectory-major (the order load_recons_data produces).

    Returns (blur, sensor_idx) as torch tensors, or (None, None) when
    --block_size is 0 so callers can keep their normal path.

    sensor_idx is (N, n_keep) — already expanded per sample and TRUNCATED to a
    common n_keep across trajectories, because different holes kill different
    numbers of sensors and DPS needs a rectangular index array. The truncation
    drops a few extra sensors from the luckier trajectories; the order of
    idx_lst is arbitrary so this is unbiased, and the count is logged.
    """
    import torch

    size = int(getattr(args, "block_size", 0) or 0)
    if size <= 0:
        return None, None

    seed = int(getattr(args, "block_seed", 0) or 0)
    pos = getattr(args, "block_pos", "") or None
    gt = ref_data.numpy() if hasattr(ref_data, "numpy") else np.asarray(ref_data)
    h, w = gt.shape[-2], gt.shape[-1]

    with np.load(config.data.sample_data_dir, allow_pickle=True) as f:
        idx_per_traj = f[idx_kw][-4:].astype(np.int64)      # test trajectories
    n_traj = idx_per_traj.shape[0]
    per_traj = gt.shape[0] // n_traj

    if log is not None:
        log(f"DEAD BLOCK: {size}x{size} unobserved square, position="
            f"{pos or f'random (seed {seed})'}; rebuilding from surviving sensors")
    blur, surviving = apply_dead_block(gt, idx_per_traj, size, seed, per_traj,
                                       h, w, log=log, pos=pos)

    n_keep = min(len(s) for s in surviving)
    stacked = np.stack([s[:n_keep] for s in surviving])      # (n_traj, n_keep)
    if log is not None:
        log(f"  sensors per trajectory truncated to a common {n_keep} "
            f"(from {[len(s) for s in surviving]}) so the index array is rectangular")
    sensor_idx = np.repeat(stacked, per_traj, axis=0)        # (N, n_keep)

    return (torch.as_tensor(blur, dtype=torch.float32),
            torch.as_tensor(sensor_idx, dtype=torch.long))


def self_test(gt, idx_per_traj, u3232, per_traj, h=256, w=256):
    """With size=0 the construction must reproduce the shipped u3232 exactly.

    Run this once on the cluster before trusting any dead-block number: it
    validates the fill against data that was made the same way.
    """
    blur, _ = apply_dead_block(gt, idx_per_traj, 0, 0, per_traj, h, w)
    diff = float(np.abs(blur - np.asarray(u3232)).max())
    return diff
