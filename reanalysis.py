"""Re-analysis: use each method's reconstruction as an INITIAL CONDITION for the
governing equations and measure how fast the simulated trajectory diverges from
the stored truth.

This is the usability test the image metrics cannot give (PCSAGAN's re-analysis
idea; the deployability review's 'derived engineering quantities' row): a field
can score well pointwise yet be useless as a simulation restart if its small
scales are wrong -- the solver amplifies exactly what MSE forgives.

Solver: pseudo-spectral 2D vorticity equation, mirroring voriticity_residual's
formulation exactly (same wavenumber layout, forcing f = -4cos(4y) on the column
axis, drag 0.1, Re = 1000), RK4 in spectral space with 2/3 dealiasing.

VALIDATION IS BUILT IN and runs first: consecutive stored truth frames are
dt = 1/32 apart, so the solver must advance truth frame j to ~ truth frame j+1.
The truth-IC control curve is the numerical floor -- every reconstruction is
judged by how much faster it diverges than that floor, so solver imperfection
cancels to first order (all ICs share it).

    python reanalysis.py            # validation + experiment + plot + table

Laptop-only: numpy on local arrays, no torch, no cluster.
"""

import glob
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# physics, matching voriticity_residual exactly
N = 256
RE = 1000.0
NU = 1.0 / RE
DRAG = 0.1
DT_FRAME = 1.0 / 32.0        # spacing of stored frames
SUBSTEPS = 8                 # solver dt = DT_FRAME / SUBSTEPS

E = "experiments/kmflow_re1000_rs256_ddim_conditional_new"
RUNS = {                     # label -> (folder, sample file)
    "truth IC (numerical floor)": None,
    "baseline": (f"{E}/guided_recons_u3232_t400_r20_w0.0", "sample_arr_run_0_it0.npy"),
    "DPS":      (f"{E}/dps_guided_recons_u3232_t400_r20_w0.0_z3.0", "sample_arr_run_0_it0.npy"),
    "SI":       (f"{E}/si_guided_recons_u3232_t400_r20_w0.0", "sample_arr_run_0_it0.npy"),
}
HORIZON = 16                                  # frames to integrate (= 0.5 time units)
STARTS_PER_TRAJ = (50, 150, 250)              # within-trajectory start indices
FRAMES_PER_TRAJ = 318                         # 4 test trajectories x 318 frames


# ---------------------------------------------------------------------------
# spectral machinery (wavenumber layout copied from voriticity_residual:
# k_x varies along axis -2, k_y along axis -1, forcing depends on the -1 axis)
k1 = np.fft.fftfreq(N) * N
KX = k1.reshape(N, 1).repeat(N, axis=1)       # varies along axis -2
KY = k1.reshape(1, N).repeat(N, axis=0)       # varies along axis -1
LAP = KX ** 2 + KY ** 2
LAP_SAFE = LAP.copy()
LAP_SAFE[0, 0] = 1.0
DEALIAS = (np.abs(KX) <= N // 3) & (np.abs(KY) <= N // 3)   # 2/3 rule

x = np.linspace(0, 2 * np.pi, N + 1)[:-1]
Y = np.tile(x, (N, 1))                        # Y[i, j] = x[j]: column axis, as verified
F_H = np.fft.fft2(-4.0 * np.cos(4.0 * Y))    # forcing, in spectral space once


def rhs(w_h):
    """d(w_h)/dt for the vorticity equation, spectral in/out, dealiased."""
    psi_h = w_h / LAP_SAFE
    u = np.fft.ifft2(1j * KY * psi_h).real
    v = np.fft.ifft2(-1j * KX * psi_h).real
    wx = np.fft.ifft2(1j * KX * w_h).real
    wy = np.fft.ifft2(1j * KY * w_h).real
    adv_h = np.fft.fft2(u * wx + v * wy) * DEALIAS
    return -adv_h - NU * LAP * w_h - DRAG * w_h + F_H


def advance(w, n_frames):
    """Advance a physical-space field by n_frames stored-frame intervals.
    Returns the field after EACH frame interval (list of length n_frames)."""
    w_h = np.fft.fft2(w.astype(np.float64))
    dt = DT_FRAME / SUBSTEPS
    out = []
    for _ in range(n_frames):
        for _ in range(SUBSTEPS):
            k1_ = rhs(w_h)
            k2_ = rhs(w_h + 0.5 * dt * k1_)
            k3_ = rhs(w_h + 0.5 * dt * k2_)
            k4_ = rhs(w_h + dt * k3_)
            w_h = w_h + (dt / 6.0) * (k1_ + 2 * k2_ + 2 * k3_ + k4_)
        out.append(np.fft.ifft2(w_h).real)
    return out


# ---------------------------------------------------------------------------
def load_sequence(folder, fn):
    """All 1272 frames in TRUE temporal order. sorted(glob) is lexicographic
    (batch0, batch1, batch10, ...), which scrambles time -- sort numerically."""
    files = glob.glob(os.path.join(folder, "sample_batch*", fn))
    files.sort(key=lambda p: int(re.search(r"sample_batch(\d+)", p).group(1)))
    return np.concatenate([np.load(f) for f in files]).astype(np.float64)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ref = load_sequence(f"{E}/guided_recons_u3232_t400_r20_w0.0", "reference_arr.npy")
    print(f"truth sequence: {ref.shape}")

    # ---- solver validation: does truth frame j advance to ~ truth frame j+1?
    print("\n== solver validation (truth frame j -> j+1, one frame interval) ==")
    errs, pers = [], []
    for j in (60, 400, 800, 1100):                     # spread across trajectories
        pred = advance(ref[j], 1)[0]
        errs.append(rel_l2(pred, ref[j + 1]))
        pers.append(rel_l2(ref[j], ref[j + 1]))        # persistence = do nothing
    print(f"  solver one-frame rel error : {np.mean(errs):.4f}  (per start: "
          + ", ".join(f"{e:.4f}" for e in errs) + ")")
    print(f"  persistence (no solver)    : {np.mean(pers):.4f}")
    if np.mean(errs) > 0.5 * np.mean(pers):
        print("  !! solver adds little skill over persistence -- treat the "
              "experiment as comparative only")
    else:
        print("  -> solver reproduces the stored dynamics; trajectories are "
              "trustworthy over short horizons")

    # ---- the experiment
    starts = [t * FRAMES_PER_TRAJ + s for t in range(4) for s in STARTS_PER_TRAJ]
    curves = {}
    for label, src in RUNS.items():
        seq = ref if src is None else load_sequence(*src)
        per_start = []
        for s in starts:
            traj = advance(seq[s], HORIZON)
            per_start.append([rel_l2(traj[h], ref[s + 1 + h]) for h in range(HORIZON)])
        curves[label] = np.array(per_start)            # (n_starts, HORIZON)
        m = curves[label].mean(axis=0)
        print(f"\n{label}: IC error {rel_l2(seq[starts[0]], ref[starts[0]]):.3f} "
              f"(first window)")
        print("  mean rel-L2 vs truth at +1/+4/+8/+16 frames: "
              f"{m[0]:.3f} / {m[3]:.3f} / {m[7]:.3f} / {m[15]:.3f}")

    # ---- table + plot
    print("\n== re-analysis: divergence from the true trajectory ==")
    print(f"{'IC':30s} {'+1':>7s} {'+4':>7s} {'+8':>7s} {'+16 frames':>11s}")
    print("-" * 60)
    for label, c in curves.items():
        m = c.mean(axis=0)
        print(f"{label:30s} {m[0]:7.3f} {m[3]:7.3f} {m[7]:7.3f} {m[15]:11.3f}")
    floor = curves["truth IC (numerical floor)"].mean(axis=0)
    print("\nexcess over the numerical floor at +8 frames "
          "(how much worse than a perfect IC):")
    for label, c in curves.items():
        if label.startswith("truth"):
            continue
        print(f"  {label:12s} {c.mean(axis=0)[7] / floor[7]:6.1f}x the floor")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    t_ax = (np.arange(HORIZON) + 1) * DT_FRAME
    for label, c in curves.items():
        m, sd = c.mean(axis=0), c.std(axis=0)
        ls = "--" if label.startswith("truth") else "-"
        ax.plot(t_ax, m, ls, label=label, linewidth=2)
        ax.fill_between(t_ax, m - sd, m + sd, alpha=0.12)
    ax.set_xlabel("integration time after restart (flow units)")
    ax.set_ylabel("relative L2 error vs true trajectory")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Re-analysis: reconstructions as simulation initial conditions\n"
                 f"(mean over {len(starts)} restarts across the 4 test trajectories)")
    fig.tight_layout()
    fig.savefig("reanalysis_divergence.png", dpi=150)
    print("\nwrote reanalysis_divergence.png")


if __name__ == "__main__":
    main()
