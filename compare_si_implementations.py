"""Compare THIS project's SI port against the source implementation of Schiodt et al.

The two solve different problems and their raw errors are NOT comparable: this
project reconstructs 256^2 VORTICITY from 1024 point sensors (1.6% coverage),
while the paper reconstructs 128^2 VELOCITY from a low-pass filter whose input
already retains most of the field. Putting MSE next to MSE would be meaningless.

What IS comparable is a set of dimensionless ratios to each run's own ground
truth, computed with ONE set of definitions applied to both:

  Z%       enstrophy <w^2> relative to truth. In 2D, dissipation = nu*<w^2>, so
           this is also the dissipation-rate ratio the paper reports.
  std%     field standard deviation relative to truth.
  e_spec   integrated relative spectral error, sum|E-E_ref| / sum E_ref.
  closed   fraction of the input->truth gap the reconstruction closes, per metric.
           This is the fairest single number: it normalises out how much the two
           measurement operators destroyed in the first place.

The paper's velocity fields are converted to vorticity spectrally so that both
sides go through the SAME compute_ke_spectrum used everywhere else in this work.

    python compare_si_implementations.py
"""

import glob
import os
import sys
import types

import numpy as np

from metrics import compute_ke_spectrum

E = "experiments/kmflow_re1000_rs256_ddim_conditional_new"
MINE_BASE = f"{E}/guided_recons_u3232_t400_r20_w0.0"          # holds input_arr + reference_arr
MINE_SI = f"{E}/si_guided_recons_u3232_t400_r20_w0.0"
PAPER = "paper_repro_data"


# --------------------------------------------------------------------------
def load_paper():
    """Their (400,2,128,128) velocity tensors -> vorticity via a spectral curl."""
    import torch
    mod = types.ModuleType("dataset_utils")          # stand in for their Dataset class

    class _Stub:
        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {"_state": s})

    mod.__getattr__ = lambda n: (_ for _ in ()).throw(AttributeError(n)) \
        if n.startswith("__") else type(n, (_Stub,), {})
    sys.modules["dataset_utils"] = mod

    ds = torch.load(f"{PAPER}/SR_test_128.pt", map_location="cpu", weights_only=False)
    pred = torch.load(f"{PAPER}/gpu_sr_ts100.pt", map_location="cpu", weights_only=False)
    return (curl(ds.x0.numpy()), curl(ds.x1.numpy()), curl(pred.numpy()))


def curl(uv):
    """Vorticity from a (N,2,H,W) velocity field, spectrally, on a 2*pi domain.

    Same integer-wavenumber layout as voriticity_residual, so the vorticity that
    comes out is on the same footing as this project's native fields.
    """
    n, _, h, w = uv.shape
    kx = np.fft.fftfreq(w, d=1.0 / w).reshape(1, 1, w)
    ky = np.fft.fftfreq(h, d=1.0 / h).reshape(1, h, 1)
    u_h = np.fft.fft2(uv[:, 0], axes=(-2, -1))
    v_h = np.fft.fft2(uv[:, 1], axes=(-2, -1))
    w_h = 1j * kx * v_h - 1j * ky * u_h
    return np.real(np.fft.ifft2(w_h, axes=(-2, -1)))


def load_mine():
    """One representative field per batch, matching what metrics.py's spectra use."""
    def stack(folder, fn):
        out = []
        for f in sorted(glob.glob(os.path.join(folder, "sample_batch*", fn))):
            a = np.load(f)
            out.append(a[-1] if a.ndim == 3 else a)
        return np.asarray(out, dtype=np.float64)
    return (stack(MINE_BASE, "input_arr.npy"),
            stack(MINE_BASE, "reference_arr.npy"),
            stack(MINE_SI, "sample_arr_run_0_it0.npy"))


# --------------------------------------------------------------------------
def spectrum(fields):
    tot, k = None, None
    for f in fields:
        k, ek = compute_ke_spectrum(f)
        tot = ek if tot is None else tot + ek
    return k, tot / len(fields)


def summarise(name, x0, x1, xp):
    """All dimensionless, all relative to that run's own truth."""
    z_t = float((x1 ** 2).mean())
    z0, zp = float((x0 ** 2).mean()) / z_t, float((xp ** 2).mean()) / z_t
    s_t = float(x1.std())
    s0, sp = float(x0.std()) / s_t, float(xp.std()) / s_t

    # normalised field error: rms(x - truth) / std(truth). Dimensionless, so it
    # survives the change of field and resolution between the two runs.
    n0 = float(np.sqrt(((x0 - x1) ** 2).mean())) / s_t
    npd = float(np.sqrt(((xp - x1) ** 2).mean())) / s_t
    c0 = float(np.corrcoef(x0.ravel(), x1.ravel())[0, 1])
    cp = float(np.corrcoef(xp.ravel(), x1.ravel())[0, 1])

    _, e_t = spectrum(x1)
    _, e_0 = spectrum(x0)
    _, e_p = spectrum(xp)
    es0 = float(np.abs(e_0 - e_t).sum() / e_t.sum())
    esp = float(np.abs(e_p - e_t).sum() / e_t.sum())

    # Fraction of the input->truth gap closed. Only meaningful where the input is
    # actually deficient: a Voronoi-filled sensor field has ~the right enstrophy
    # (its block discontinuities supply the missing variance for the wrong
    # reason), so that denominator collapses and the ratio is uninformative.
    cz = (zp - z0) / (1.0 - z0) if abs(1.0 - z0) > 0.02 else float("nan")
    ce = (es0 - esp) / es0 if es0 > 0 else float("nan")
    cn = (n0 - npd) / n0 if n0 > 0 else float("nan")

    print(f"\n{name}")
    print(f"  {'':24s} {'input':>10s} {'reconstr.':>10s} {'truth':>8s}")
    print(f"  {'norm. field error':24s} {n0:10.3f} {npd:10.3f} {0.0:8.3f}")
    print(f"  {'correlation':24s} {c0:10.4f} {cp:10.4f} {1.0:8.3f}")
    print(f"  {'enstrophy / dissipation':24s} {z0:10.3f} {zp:10.3f} {1.0:8.3f}")
    print(f"  {'std ratio':24s} {s0:10.3f} {sp:10.3f} {1.0:8.3f}")
    print(f"  {'e_spec':24s} {es0:10.3f} {esp:10.3f} {0.0:8.3f}")
    cz_s = f"{100*cz:6.1f}%" if np.isfinite(cz) else "   n/a"
    print(f"  gap closed — field {100*cn:6.1f}%   spectral {100*ce:6.1f}%   enstrophy {cz_s}")
    if not np.isfinite(cz):
        print("    (enstrophy gap n/a: the input already matches truth's variance,"
              " so there is no deficit to close)")
    return {"z0": z0, "zp": zp, "s0": s0, "sp": sp, "es0": es0, "esp": esp,
            "n0": n0, "np": npd, "c0": c0, "cp": cp, "cz": cz, "ce": ce, "cn": cn}


def main():
    print("Loading the paper's tensors (velocity -> vorticity via spectral curl)...")
    p0, p1, pp = load_paper()
    print(f"  paper: {p0.shape}  vorticity std truth={p1.std():.3f}")

    print("Loading this project's fields...")
    m0, m1, mp = load_mine()
    print(f"  mine:  {m0.shape}  vorticity std truth={m1.std():.3f}")

    a = summarise("PAPER — Schiodt et al., 128^2 velocity, low-pass input", p0, p1, pp)
    b = summarise("MINE  — this port, 256^2 vorticity, 1024 sensors (1.6%)", m0, m1, mp)

    print("\n" + "=" * 70)
    print("SIDE BY SIDE (dimensionless, each against its OWN truth)")
    print(f"  {'':28s} {'paper':>10s} {'mine':>10s}")
    for lab, k in (("input norm. field error", "n0"), ("output norm. field error", "np"),
                   ("input correlation", "c0"), ("output correlation", "cp"),
                   ("input e_spec", "es0"), ("output e_spec", "esp"),
                   ("output enstrophy ratio", "zp")):
        print(f"  {lab:28s} {a[k]:10.4f} {b[k]:10.4f}")
    print(f"  {'gap closed, field':28s} {100*a['cn']:9.1f}% {100*b['cn']:9.1f}%")
    print(f"  {'gap closed, spectral':28s} {100*a['ce']:9.1f}% {100*b['ce']:9.1f}%")
    print("\n  Read the 'gap closed' rows: they normalise out how much each")
    print("  measurement operator destroyed, which is what differs between tasks.")


if __name__ == "__main__":
    main()
