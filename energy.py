import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from collections import defaultdict
import os
import glob

from metrics import compute_ke_spectrum
from metrics import collect_and_average_data, parse_entry, method_dir, shade


EXPERIMENT_FOLDER = "kmflow_re1000_rs256_ddim_conditional_new"
DATA_KW = "u3232"
T = 400
R = 20
W = 0.0

ZETA = 3.0        
SI_LAMBDA = 0.01  
SI_W = 3.0        

METHOD_CONFIG = {
    "baseline": {
        "prefix": "", "takes_value": False, "default": None,
        "suffix": lambda v: "",
        "label":  lambda v: "Baseline (physics-guided)",
        "color": "red", "linestyle": "-",
    },
    "dps": {
        "prefix": "dps_", "takes_value": True, "default": ZETA,
        "suffix": lambda v: f"_z{v}",
        "label":  lambda v: f"DPS (zeta={v})",
        "color": "green", "linestyle": "-",
    },
    "si": {
        "prefix": "si_", "takes_value": False, "default": None,
        "suffix": lambda v: "",
        "label":  lambda v: "Stochastic interpolant",
        "color": "darkorange", "linestyle": "-",
    },
    "si_blind": {
        "prefix": "si_", "takes_value": False, "default": None,
        "suffix": lambda v: "_blind",
        "label":  lambda v: "SI (blind, robust)",
        "color": "teal", "linestyle": "-",
    },
    "si_linear": {
        "prefix": "si_", "takes_value": True, "default": SI_LAMBDA,
        "suffix": lambda v: f"_linear_lam{v}",
        "label":  lambda v: f"SI + linear physics (lam={v})",
        "color": "purple", "linestyle": "-",
    },
    "si_learned": {
        "prefix": "si_", "takes_value": True, "default": SI_W,
        "suffix": lambda v: f"_learned_w{v}",
        "label":  lambda v: f"SI + learned physics (w={v})",
        "color": "saddlebrown", "linestyle": "-",
    },
}
REFERENCE_STYLE = {"label": "Reference", "color": "mediumblue", "linestyle": "--"}




QUANTITY = "kinetic"          # "kinetic" | "enstrophy"
SHOW_DIFF = True              # add a second row of (method - reference) maps
CMAP = "inferno"              # energy maps (non-negative): bright = high energy
CMAP_DIFF = "RdBu_r"          # difference maps: red = too much, blue = too little


def _wavenumbers(N):
    k = np.fft.fftfreq(N) * N
    kx = k.reshape(N, 1)      # varies along axis -2 (rows)
    ky = k.reshape(1, N)      # varies along axis -1 (cols)
    lap = kx ** 2 + ky ** 2
    lap[0, 0] = 1.0           # avoid /0 at the mean mode
    return kx, ky, lap


def energy_density_map(directory, file_name, quantity=QUANTITY):
    """Mean spatial energy density over ALL frames in a run.

    Streams the per-batch .npy files (each (frames, 256, 256) vorticity) and
    accumulates the running mean map, so memory stays at one 256x256 array.
    Returns a (256, 256) map in physical units.
    """
    files = sorted(glob.glob(os.path.join(directory, "sample_batch*", file_name)))
    if not files:
        raise FileNotFoundError(f"No files matching {file_name} under {directory}")

    acc = None
    n_frames = 0
    kx = ky = lap = None
    for f in files:
        w = np.load(f).astype(np.float64)          # (F, 256, 256)
        if w.ndim == 2:
            w = w[None]
        F, N, _ = w.shape
        if quantity == "enstrophy":
            e = 0.5 * w ** 2
        else:                                       # kinetic energy from vorticity
            if lap is None:
                kx, ky, lap = _wavenumbers(N)
            w_h = np.fft.fft2(w, axes=(-2, -1))
            psi_h = w_h / lap                       # streamfunction: -lap psi = w
            u = np.fft.ifft2(1j * ky * psi_h, axes=(-2, -1)).real     # u =  d psi / dy
            v = np.fft.ifft2(-1j * kx * psi_h, axes=(-2, -1)).real    # v = -d psi / dx
            e = 0.5 * (u ** 2 + v ** 2)
        acc = e.sum(axis=0) if acc is None else acc + e.sum(axis=0)
        n_frames += F
    return acc / n_frames


def plot_energy(methods_to_plot):
    """Spatial energy maps: 'where is the energy?' for each selected config,
    with the reference alongside. Reference is ALWAYS included.

    Each entry is 'method' or ('method', value) -- same selection style as
    metrics.py (e.g. ["baseline", "si", ("si_linear", 0.01)]).
    """
    # Resolve entries into (method, value), skipping unknown/missing folders.
    panels = []                                     # (label, map)
    reference_dir = None
    for entry in methods_to_plot:
        name = entry[0] if isinstance(entry, (tuple, list)) else entry
        if name not in METHOD_CONFIG:
            print(f"Unknown method '{name}'. Skipping.")
            continue
        method, value = parse_entry(entry)
        d = method_dir(method, value)
        if not os.path.isdir(d):
            print(f"Warning: folder for '{method}' (value={value}) not found: {d}. Skipping.")
            continue
        print(f"Computing {QUANTITY} map for {method} (value={value}) ...")
        panels.append((METHOD_CONFIG[method]["label"](value),
                       energy_density_map(d, "sample_arr_run_0_it0.npy")))
        if reference_dir is None:
            reference_dir = d

    if not panels:
        print("No valid method folders found -- nothing to plot.")
        return

    print("Computing reference map ...")
    ref_map = energy_density_map(reference_dir, "reference_arr.npy")
    panels.append((REFERENCE_STYLE["label"], ref_map))

    qname = "Kinetic energy density" if QUANTITY == "kinetic" else "Enstrophy density"
    ncol = len(panels)
    nrow = 2 if SHOW_DIFF else 1

    # Shared scale across the energy row so brightness is comparable between
    # methods; clip at the 99.5th percentile of the reference to keep hotspots
    # from washing everything else out.
    vmax = np.percentile(ref_map, 99.5)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow), squeeze=False)

    for c, (label, emap) in enumerate(panels):
        ax = axes[0][c]
        im = ax.imshow(emap, cmap=CMAP, vmin=0, vmax=vmax, origin="lower")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02,
                 label=f"{qname}  (shared scale)")

    if SHOW_DIFF:
        dmax = np.percentile(np.abs(np.stack([m for _, m in panels[:-1]]) - ref_map), 99.5)
        for c, (label, emap) in enumerate(panels):
            ax = axes[1][c]
            if label == REFERENCE_STYLE["label"]:
                ax.axis("off")
                ax.text(0.5, 0.5, "(reference)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue
            im2 = ax.imshow(emap - ref_map, cmap=CMAP_DIFF, vmin=-dmax, vmax=dmax,
                            origin="lower")
            ax.set_title(f"{label} − reference", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im2, ax=axes[1].tolist(), fraction=0.046, pad=0.02,
                     label="energy error  (red = excess, blue = deficit)")

    fig.suptitle(f"{qname}: where is the energy?  (mean over test frames)",
                 fontsize=12, y=1.0)

    tag = "_".join(m if v is None else f"{m}{v}"
                   for m, v in (parse_entry(e) for e in methods_to_plot))
    save_target = os.path.join("experiments", EXPERIMENT_FOLDER,
                               f"energy_map_{QUANTITY}_{tag}.png")
    plt.savefig(save_target, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to: {save_target}")
    plt.show()


if __name__ == "__main__":
    # Which configs to map. Reference is ALWAYS included. Same entry style as
    # metrics.py: "method" or ("method", value).
    #   ["baseline", "si"]                  -> baseline vs SI vs reference
    #   ["si", ("si_linear", 0.01)]         -> does physics guidance relocate energy?
    METHODS_TO_PLOT = ["baseline", "dps", "si"]

    plot_energy(METHODS_TO_PLOT)