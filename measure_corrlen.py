#!/usr/bin/env python
"""Measure the vorticity correlation length that the whole depth axis rests on.

    python measure_corrlen.py [--root DIR] [--max_r 64]

WHY THIS EXISTS. `corrlen = 14 px` was a hard-coded constant with no script
behind it, and it is load-bearing: every depth in the thesis is quoted in units
of it, and the central claim -- that the transport formulation stops paying
beyond one correlation length -- is bracketed by exactly two configurations, at
0.71 and 1.14 of it. If the constant were 20 px those become 0.50 and 0.80 and
the claim is false. So it needs to be measurable, not asserted.

MEASURED 2026-09-02: 14.56 px over 1,272 frames. The thesis uses 14, which is
within 4% and leaves the crossing straddling one correlation length (10 px =
0.69, 16 px = 1.10).

METHOD. The two-point autocorrelation via FFT, which is exact on a periodic
domain and O(N log N) rather than the O(N^2) of an explicit pair sum:

    C(r) = < w(x) w(x+r) >   normalised so C(0) = 1,

radially averaged, and read where it falls to 1/e. That is the separation at
which an exponential decay has dropped by one decay length, and it is the
convention of Pope, Turbulent Flows, Sec. 6.3.

THE MEAN IS REMOVED PER FRAME before transforming. Leaving it in adds a constant
to C(r) that never decays, so the curve flattens above 1/e and the length comes
out unbounded -- a failure that looks like a physics result rather than a bug.

CAVEAT ON THE DATA. This reads whatever reference fields are under --root. The
thesis says the length is measured on the training split; the frames used here
are Re 1000 vorticity at 256^2 from the cross-Re run directory, which is the
same flow but not provably the identical split. Re-run it against the training
frames on the cluster to close that gap.
"""
import argparse
import glob
import os
import sys

import numpy as np

DEFAULT_ROOT = "experiments/cross_re1000"


def profile(files, max_r):
    """Radially averaged, normalised autocorrelation out to max_r pixels."""
    acc, n = None, 0
    for f in files:
        a = np.load(f).astype(np.float64)
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
        for fr in a:
            fr = fr - fr.mean()                 # see docstring: not optional
            F = np.fft.rfft2(fr)
            acc = (np.fft.irfft2(F * np.conj(F), s=fr.shape) if acc is None
                   else acc + np.fft.irfft2(F * np.conj(F), s=fr.shape))
            n += 1
    if not n:
        sys.exit("no frames found")
    acc /= n
    acc /= acc[0, 0]
    h, w = acc.shape
    yy = np.minimum(np.arange(h), h - np.arange(h))
    xx = np.minimum(np.arange(w), w - np.arange(w))
    rb = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2).astype(int)
    return np.array([acc[rb == r].mean() for r in range(max_r)]), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--max_r", type=int, default=64)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "*", "sample_batch*",
                                          "reference_arr.npy")))
    if not files:
        sys.exit("no reference_arr.npy under %s" % a.root)

    prof, n = profile(files, a.max_r)
    inv_e = float(np.exp(-1.0))
    below = np.where(prof < inv_e)[0]
    if not len(below):
        sys.exit("C(r) never falls below 1/e within %d px" % a.max_r)

    r1 = int(below[0])
    r0 = r1 - 1
    lc = r0 + (prof[r0] - inv_e) / (prof[r0] - prof[r1])   # linear interpolation

    print("frames averaged : %d  (%d files under %s)" % (n, len(files), a.root))
    print("C(r) crosses 1/e = %.4f between r = %d and r = %d" % (inv_e, r0, r1))
    print("correlation length = %.2f px" % lc)
    print()
    print("depths used in the thesis, in units of this length:")
    for d in (4, 5, 8, 10, 16, 32, 64):
        print("   %2d px = %.2f lc" % (d, d / lc))
    print()
    print("The crossing sits between the 10 px and 16 px configurations, so it")
    print("straddles one correlation length as long as lc lies between 10 and 16.")


if __name__ == "__main__":
    main()
