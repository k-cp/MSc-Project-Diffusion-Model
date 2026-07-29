"""Distributional metrics from the SI paper (Schiodt et al. 2026, Table 1):
KL divergence and Wasserstein-1 distance between each method's POINTWISE value
distribution and the ground truth's, for vorticity w and kinetic energy E.

This is the paper's own evaluation protocol applied to our runs, so the thesis
can state whether their ordering (SI best by a wide margin) replicates in our
setting. Absolute numbers are NOT comparable across papers (different variable,
degradation, normalisation) -- only the ordering and the ratios are.

w is our native variable. E = u^2 + v^2 is derived per frame from the
streamfunction (u_h = i k_y psi_h, v_h = -i k_x psi_h, psi_h = w_h / k^2),
reusing the exact wavenumber layout validated in reanalysis.py.

Distributions are pooled pointwise over the FULL test set (1272 x 256^2 values
per method), estimated on a common 512-bin histogram. W1 for 1D distributions
is exact from CDFs: W1 = sum |CDF_p - CDF_q| * dx.

    python distribution_metrics.py         # laptop-only, ~2 min
"""

import numpy as np

from metrics import normalize, spec_to_folder, spec_to_sample_file, _load_full
from reanalysis import KX, KY, LAP_SAFE   # validated wavenumber layout

BINS = 512
METHODS = [
    ("input x0 (u3232)", "baseline", "input_arr.npy"),      # measurement itself
    ("baseline",          "baseline", None),
    ("DPS",               ("dps", 3.0), None),
    ("SI",                "si", None),
]


def kinetic_energy(w_seq):
    """Pointwise E = u^2 + v^2 for a (T, 256, 256) vorticity sequence."""
    out = np.empty_like(w_seq)
    for i, w in enumerate(w_seq):
        w_h = np.fft.fft2(w)
        psi_h = w_h / LAP_SAFE
        u = np.fft.ifft2(1j * KY * psi_h).real
        v = np.fft.ifft2(-1j * KX * psi_h).real
        out[i] = u * u + v * v
    return out


def kl_w1(sample, ref, lo, hi):
    """KL(ref || sample) and exact 1-D W1 from a common histogram."""
    edges = np.linspace(lo, hi, BINS + 1)
    dx = edges[1] - edges[0]
    p, _ = np.histogram(ref, bins=edges, density=True)
    q, _ = np.histogram(sample, bins=edges, density=True)
    p, q = p + 1e-12, q + 1e-12
    p, q = p / (p.sum() * dx), q / (q.sum() * dx)
    kl = float(np.sum(p * np.log(p / q)) * dx)
    w1 = float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * dx * dx)
    return kl, w1


def main():
    ref_dir = spec_to_folder(normalize("baseline"))
    ref = _load_full(ref_dir, "reference_arr.npy")
    print(f"reference: {ref.shape}")
    ref_E = kinetic_energy(ref)

    # common histogram ranges from the reference, padded so no method clips
    wlo, whi = 1.5 * ref.min(), 1.5 * ref.max()
    elo, ehi = 0.0, 1.5 * ref_E.max()

    print(f"\n{'method':22s} {'KL(w)':>9s} {'W1(w)':>9s} {'KL(E)':>9s} {'W1(E)':>9s}")
    print("-" * 64)
    print(f"{'reference':22s} {0.0:9.5f} {0.0:9.4f} {0.0:9.5f} {0.0:9.4f}")
    rows = []
    for label, spec, override in METHODS:
        s = normalize(spec)
        fn = override or spec_to_sample_file(s)
        x = _load_full(spec_to_folder(s), fn)
        if x is None:
            print(f"{label:22s} {'-- (arrays not local)':>40s}")
            continue
        kw, ww = kl_w1(x, ref, wlo, whi)
        xE = kinetic_energy(x)
        ke, we = kl_w1(xE, ref_E, elo, ehi)
        rows.append((label, kw, ww, ke, we))
        print(f"{label:22s} {kw:9.5f} {ww:9.4f} {ke:9.5f} {we:9.4f}")

    base = next((r for r in rows if r[0] == "baseline"), None)
    si = next((r for r in rows if r[0] == "SI"), None)
    if base and si:
        print(f"\nSI improvement over the diffusion baseline (their Table-1 style):")
        for name, i in (("KL(w)", 1), ("W1(w)", 2), ("KL(E)", 3), ("W1(E)", 4)):
            print(f"  {name}: {base[i] / max(si[i], 1e-12):6.1f}x")
    print("\n(their Table 1 finds SI 4-10x better than paired DM/FM on these metrics;")
    print(" the comparable statement here is the ratio vs OUR diffusion baselines.)")


if __name__ == "__main__":
    main()
