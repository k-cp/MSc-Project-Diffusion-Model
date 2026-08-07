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
  1. the input field — the hole is refilled from the nearest SURVIVING sensor,
     so it is extrapolated from its boundary exactly as a real dead patch would
     be. Everything outside the hole is left untouched;
  2. DPS's sensor index set — DPS gathers ground truth at idx_lst directly
     (posterior_sampling.py), so leaving those indices in place would let it read
     the true field inside the hole.

WHY THE HOLE IS PATCHED RATHER THAN THE FIELD REBUILT. The first version of this
module rebuilt the whole measurement from the surviving sensors. Its validation
gate failed: the result disagreed with the shipped u3232 on 4.8% of pixels
(max 18.0). Diagnosis — u3232 IS a nearest-neighbour fill (100% of its pixels
equal some sensor's value exactly, and it equals ground truth at the sensors to
0.0), but its tie-breaking for pixels equidistant between two sensors differs
from scipy's distance_transform_edt. Both fills are valid; they simply disagree
on ties. Rebuilding would therefore have perturbed the entire field and made
these runs incomparable with every other result in the project. Patching only
the hole keeps the rest bit-identical.

SELF-CHECK. Invariants, verified by verify_dead_block.py against the real data
before any run: size=0 leaves the input untouched; a real block leaves
everything outside the hole untouched; the hole changes; and no surviving sensor
lies inside it.
"""

import numpy as np


def read_npz_trajectories(path, key, traj_slice, n_frames=None):
    """Read only a few trajectories out of a big .npz member, without loading it all.

    `NpzFile[key]` decompresses the WHOLE array before any slice is applied — for
    the (40, 320, 256, 256) measurement file that is ~3.4 GB and is enough to get
    a login-node process OOM-killed. Each member of an .npz is a .npy stream, so
    we can parse its header, seek to the trajectories we want (they are
    contiguous) and read just those bytes: ~21 MB for 4 trajectories x 20 frames.

    traj_slice : slice over the FIRST axis, e.g. slice(-4, None).
    n_frames   : keep only the first n frames of each trajectory (None = all).
    """
    import zipfile
    from numpy.lib import format as npformat

    with zipfile.ZipFile(path) as z:
        name = key if key.endswith(".npy") else key + ".npy"
        if name not in z.namelist():
            raise KeyError(f"{name!r} not in {path} (has {z.namelist()})")
        with z.open(name) as fh:
            version = npformat.read_magic(fh)
            shape, fortran, dtype = npformat._read_array_header(fh, version)
            if fortran:
                raise ValueError("Fortran-ordered .npy member not supported here")
            n_traj = shape[0]
            per_traj = int(np.prod(shape[1:]))
            frame = int(np.prod(shape[2:])) if len(shape) > 2 else per_traj
            keep = range(*traj_slice.indices(n_traj))
            want = n_frames if n_frames is not None else shape[1]

            out = np.empty((len(keep), want) + tuple(shape[2:]), dtype=dtype)
            pos = 0                       # elements consumed from the stream
            for i, t in enumerate(keep):
                start = t * per_traj      # first element of this trajectory
                need = want * frame
                skip = start - pos
                if skip < 0:
                    raise ValueError("trajectory indices must be increasing")
                _consume(fh, skip * dtype.itemsize)
                buf = fh.read(need * dtype.itemsize)
                out[i] = np.frombuffer(buf, dtype=dtype, count=need).reshape(
                    (want,) + tuple(shape[2:]))
                pos = start + need
    return out


def _consume(fh, nbytes, chunk=1 << 22):
    """Skip forward in a zip member stream without holding it in memory."""
    while nbytes > 0:
        got = fh.read(min(chunk, nbytes))
        if not got:
            raise EOFError("unexpected end of .npy member")
        nbytes -= len(got)


def parse_size(size, h=256, w=256):
    """Normalise a --block_size value to (block_h, block_w) in pixels.

    Accepts, so the hole's AREA and ASPECT are both controllable:
      64          a 64x64 square           (the original form, unchanged)
      "64"        same
      "16x96"     a 16-tall, 96-wide slab  -> HxW, same area as a 39x39 square
      "0"/0/None  disabled -> (0, 0)
      "6.25%"     a square covering that fraction of the field's AREA
      (16, 96)    an explicit pair

    Returns (0, 0) when disabled, so callers can keep testing `if bh <= 0`.
    """
    if size is None:
        return 0, 0
    if isinstance(size, (tuple, list)):
        bh, bw = int(size[0]), int(size[1])
    else:
        s = str(size).strip().lower()
        if not s or s == "0":
            return 0, 0
        if s.endswith("%"):
            frac = float(s[:-1]) / 100.0
            if not 0.0 < frac <= 1.0:
                raise ValueError(f"--block_size {size!r}: percentage must be in (0, 100]")
            side = int(round((frac * h * w) ** 0.5))
            bh = bw = max(1, side)
        elif "x" in s:
            a, _, b = s.partition("x")
            bh, bw = int(a), int(b)
        else:
            bh = bw = int(float(s))
    if bh <= 0 or bw <= 0:
        return 0, 0
    if bh > h or bw > w:
        raise ValueError(f"--block_size {size!r} -> {bh}x{bw} does not fit the {h}x{w} field")
    return bh, bw


def size_tag(bh, bw):
    """Folder/log tag: '64' for a square, '16x96' for a rectangle."""
    return f"{bh}" if bh == bw else f"{bh}x{bw}"


def block_for_trajectory(size, seed, traj, h=256, w=256, pos=None):
    """Top-left corner of the dead block for one trajectory.

    One block per TRAJECTORY, not per frame: a sensor patch fails and stays
    failed, so the hole must sit still while the flow evolves through it.

    size is anything parse_size accepts — an int for a square, "HxW" for a
    rectangle, or "P%" for a target area fraction.

    pos controls WHERE:
      None / ""      random per trajectory, seeded on (seed, traj) — different
                     independent holes, which averages over flow features and is
                     the right default for a quantitative result.
      "center"       centred in the field, identical for every trajectory.
      "y,x"          explicit top-left corner, identical for every trajectory —
                     use this when a figure must show the same region every time.
    """
    bh, bw = parse_size(size, h, w)
    if bh <= 0:
        return None
    if pos:
        if str(pos).strip() == "center":
            return (h - bh) // 2, (w - bw) // 2
        y, _, x = str(pos).partition(",")
        y, x = int(y), int(x)
        if not (0 <= y <= h - bh and 0 <= x <= w - bw):
            raise ValueError(f"--block_pos {pos!r} puts a {bh}x{bw} block outside "
                             f"the {h}x{w} field")
        return y, x
    rng = np.random.default_rng(int(seed) * 100003 + int(traj))
    return int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))


def surviving_sensors(flat_idx, block, size, h=256, w=256):
    """Drop the sensors that fall inside the dead block.

    flat_idx : (n,) flat indices into the h*w grid (one trajectory's layout).
    Returns the surviving subset, and how many died.
    """
    bh, bw = parse_size(size, h, w)
    if block is None or bh <= 0:
        return flat_idx, 0
    y0, x0 = block
    iy, ix = np.divmod(np.asarray(flat_idx), w)
    dead = (iy >= y0) & (iy < y0 + bh) & (ix >= x0) & (ix < x0 + bw)
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


def apply_dead_block(meas, idx_per_traj, size, seed, per_traj, h=256, w=256,
                     log=None, pos=None, fill="extrapolate"):
    """Kill a square of sensors in an EXISTING measurement field.

    Takes the shipped measurement (u3232) and overwrites ONLY the hole, leaving
    every pixel outside it bit-identical to the input.

    Why not rebuild the whole field from the surviving sensors? Because u3232 is
    a nearest-neighbour fill whose TIE-BREAKING differs from scipy's
    distance_transform_edt: both are valid NN fills (verified — 100% of u3232's
    pixels equal some sensor's value exactly), but they disagree on the ~4.8% of
    pixels that are equidistant between two sensors. Rebuilding would therefore
    perturb the whole field and make the dead-block runs incomparable with every
    other run in this project. Overwriting only the hole avoids that entirely.

    The hole is filled from the nearest SURVIVING sensor. Sensor values are read
    out of `meas` itself, which is exact there (u3232 equals ground truth at the
    sensors — verified, max|diff| = 0), so no ground truth is needed.

    fill controls what the model SEES inside the hole:
      "extrapolate"  nearest surviving sensor — a realistic dead-sensor patch,
                     where the region looks plausible but is unconstrained.
      "zero"         a true void. The data is zero-mean, so 0 carries no
                     information rather than misleading information: this is
                     classical inpainting, and the model must generate the
                     region rather than sharpen a wrong guess.

    meas         : (N, C, h, w) the shipped measurement, trajectory-major.
    idx_per_traj : (n_traj, n_sensors) sensor layout per trajectory.
    Returns (blur, surviving_flat); DPS must be given surviving_flat, not the
    original indices.
    """
    from scipy.ndimage import distance_transform_edt

    meas = np.asarray(meas)
    n_traj = len(idx_per_traj)
    if n_traj * per_traj != meas.shape[0]:
        raise ValueError(f"{n_traj} trajectories x {per_traj} != {meas.shape[0]} samples")

    bh, bw = parse_size(size, h, w)
    blur = meas.copy()
    surviving = []
    for t in range(n_traj):
        block = block_for_trajectory(size, seed, t, h, w, pos)
        keep, n_dead = surviving_sensors(idx_per_traj[t], block, size, h, w)
        surviving.append(keep)
        lo, hi = t * per_traj, (t + 1) * per_traj
        if block is not None:
            y0, x0 = block
            mask = np.zeros(h * w, dtype=bool)
            mask[np.asarray(keep)] = True
            if not mask.any():
                raise ValueError("the dead block covers every sensor")
            iy, ix = distance_transform_edt(~mask.reshape(h, w),
                                            return_distances=False,
                                            return_indices=True)
            if fill == "zero":
                blur[lo:hi, ..., y0:y0 + bh, x0:x0 + bw] = 0.0
            else:
                sub_y = iy[y0:y0 + bh, x0:x0 + bw]     # nearest surviving sensor
                sub_x = ix[y0:y0 + bh, x0:x0 + bw]     # for each pixel in the hole
                blur[lo:hi, ..., y0:y0 + bh, x0:x0 + bw] = \
                    meas[lo:hi][..., sub_y, sub_x]
        if log is not None:
            where = "none" if block is None else f"at (y={block[0]}, x={block[1]})"
            log(f"  trajectory {t}: block {bh}x{bw} {where}; "
                f"{n_dead}/{len(idx_per_traj[t])} sensors dead, {len(keep)} surviving")
    return blur, surviving


def build_from_config(args, config, blur_data, log=None, idx_kw="idx_lst"):
    """Whole dead-block measurement for a runner, from the flattened test set.

    blur_data : (N, C, h, w) the SHIPPED measurement (u3232) as loaded by
    load_recons_data — trajectory-major. Only the hole is overwritten; every
    other pixel is passed through unchanged, so runs stay comparable.

    Returns (blur, sensor_idx) as torch tensors, or (None, None) when
    --block_size is 0 so callers can keep their normal path.

    sensor_idx is (N, n_keep) — already expanded per sample and TRUNCATED to a
    common n_keep across trajectories, because different holes kill different
    numbers of sensors and DPS needs a rectangular index array. The truncation
    drops a few extra sensors from the luckier trajectories; the order of
    idx_lst is arbitrary so this is unbiased, and the count is logged.
    """
    import torch

    size = getattr(args, "block_size", 0)
    meas = blur_data.numpy() if hasattr(blur_data, "numpy") else np.asarray(blur_data)
    h, w = meas.shape[-2], meas.shape[-1]
    bh, bw = parse_size(size, h, w)
    if bh <= 0:
        return None, None

    seed = int(getattr(args, "block_seed", 0) or 0)
    pos = getattr(args, "block_pos", "") or None
    fill = getattr(args, "block_fill", "extrapolate") or "extrapolate"

    with np.load(config.data.sample_data_dir, allow_pickle=True) as f:
        idx_per_traj = f[idx_kw][-4:].astype(np.int64)      # test trajectories
    n_traj = idx_per_traj.shape[0]
    per_traj = meas.shape[0] // n_traj

    if log is not None:
        area = bh * bw
        log(f"DEAD BLOCK: {bh}x{bw} unobserved region "
            f"({area:,} px = {100.0 * area / (h * w):.2f}% of the field), position="
            f"{pos or f'random (seed {seed})'}, fill={fill}"
            + ("  (true void - inpainting)" if fill == "zero"
               else "  (extrapolated from surviving sensors)"))
    blur, surviving = apply_dead_block(meas, idx_per_traj, size, seed, per_traj,
                                       h, w, log=log, pos=pos, fill=fill)

    n_keep = min(len(s) for s in surviving)
    stacked = np.stack([s[:n_keep] for s in surviving])      # (n_traj, n_keep)
    if log is not None:
        log(f"  sensors per trajectory truncated to a common {n_keep} "
            f"(from {[len(s) for s in surviving]}) so the index array is rectangular")
    sensor_idx = np.repeat(stacked, per_traj, axis=0)        # (N, n_keep)

    return (torch.as_tensor(blur, dtype=torch.float32),
            torch.as_tensor(sensor_idx, dtype=torch.long))


def self_test(u3232, idx_per_traj, per_traj, size, seed, h=256, w=256, pos=None):
    """Invariants the construction must satisfy. Returns a dict of measurements.

    With size=0 the output must be the input untouched. With a real block, the
    field must be unchanged OUTSIDE the hole and changed INSIDE it, and no
    surviving sensor may lie within the hole.
    """
    u = np.asarray(u3232)
    bh, bw = parse_size(size, h, w)
    zero, _ = apply_dead_block(u, idx_per_traj, 0, 0, per_traj, h, w)
    blur, surv = apply_dead_block(u, idx_per_traj, size, seed, per_traj, h, w, pos=pos)
    b = block_for_trajectory(size, seed, 0, h, w, pos)
    y0, x0 = b
    outside = np.ones((h, w), bool)
    outside[y0:y0 + bh, x0:x0 + bw] = False
    iy, ix = np.divmod(surv[0], w)
    return {
        "size0_untouched": float(np.abs(zero - u).max()),
        "outside_untouched": float(np.abs(blur[:per_traj][..., outside]
                                          - u[:per_traj][..., outside]).max()),
        "inside_changed": float(np.abs(blur[:per_traj][..., y0:y0 + bh, x0:x0 + bw]
                                       - u[:per_traj][..., y0:y0 + bh, x0:x0 + bw]).mean()),
        "survivors_in_hole": int(((iy >= y0) & (iy < y0 + bh) &
                                  (ix >= x0) & (ix < x0 + bw)).sum()),
    }
