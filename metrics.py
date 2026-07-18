import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from collections import defaultdict
import os
import glob

def compute_ke_spectrum(vorticity_field):
    """Computes the 1D kinetic energy spectrum E(k) from a 2D vorticity field."""
    N = vorticity_field.shape[0]
    
    # 2D Fast Fourier Transform
    w_hat = np.fft.fftshift(np.fft.fft2(vorticity_field))
    
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N
    ky = np.fft.fftshift(np.fft.fftfreq(N)) * N
    KX, KY = np.meshgrid(kx, ky)
    
    K_sq = KX**2 + KY**2
    K_sq[K_sq == 0] = 1e-10
    
    E_2d = 0.5 * (np.abs(w_hat)**2) / K_sq
    E_2d[N//2, N//2] = 0 
    
    K_mag = np.sqrt(K_sq)
    K_int = np.round(K_mag).astype(int)
    
    max_k = N // 2
    E_k = np.zeros(max_k)
    for i in range(1, max_k):
        E_k[i] = np.sum(E_2d[K_int == i])
        
    return np.arange(1, max_k), E_k[1:]

def collect_and_average_data(batch_dir_pattern, file_name):
    """
    Finds all matching .npy files across batch folders and collects them.
    """
    search_path = os.path.join(batch_dir_pattern, file_name)
    file_list = glob.glob(search_path)
    
    if not file_list:
        raise FileNotFoundError(f"Could not find any files matching: {search_path}")
        
    print(f"Found {len(file_list)} batch folders containing '{file_name}'.")
    
    fields = []
    for f in file_list:
        data = np.load(f)
        # If sequence data is saved, grab the final frame
        if data.ndim == 3:
            data = data[-1]  
        fields.append(data)
        
    return fields


EXPERIMENT_FOLDER = "kmflow_re1000_rs256_ddim_conditional_new"
DATA_KW = "u3232"
T = 400
R = 20
W = 0.0
# Defaults used when an entry in METHODS_TO_PLOT gives no explicit value.
ZETA = 3.0        # DPS      --zeta
SI_LAMBDA = 0.01  # SI linear   --si_lambda
SI_W = 3.0        # SI learned  --si_w

# Per-method definition. 'suffix' and 'label' are functions of the run's value
# (zeta / lambda / w), so the same method can be plotted at several values.
# The suffix must match the folder main.py created.
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


def parse_entry(entry):
    """Accept 'si' or ('si_linear', 0.01) -> (method, value).

    A bare method name falls back to that method's default value.
    """
    if isinstance(entry, (tuple, list)):
        method, value = entry[0], entry[1]
    else:
        method, value = entry, None
    cfg = METHOD_CONFIG[method]
    if value is None and cfg["takes_value"]:
        value = cfg["default"]
    return method, value


def method_dir(method, value=None):
    """Folder for one run. The suffix carries its tag (DPS zeta, SI physics
    strength) and must match what main.py created."""
    cfg = METHOD_CONFIG[method]
    folder = f"{cfg['prefix']}guided_recons_{DATA_KW}_t{T}_r{R}_w{W}{cfg['suffix'](value)}"
    return os.path.join("experiments", EXPERIMENT_FOLDER, folder)


def shade(base_color, i, n):
    """Distinguish n runs of the same method: i=0 keeps the base colour,
    later ones are progressively lightened toward white."""
    if n <= 1:
        return base_color
    rgb = np.array(mcolors.to_rgb(base_color))
    f = 1.0 - 0.55 * (i / (n - 1))
    return tuple(1.0 - f * (1.0 - rgb))


def plot_fluid_statistics(methods_to_plot):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})

    # Build the list of curves: each selected method's reconstruction, then the
    # reference once (loaded from the first available method folder -- the
    # ground truth is identical across folders).
    # Resolve entries ('si' or ('si_linear', 0.01)) into (method, value) pairs.
    parsed = []
    for entry in methods_to_plot:
        name = entry[0] if isinstance(entry, (tuple, list)) else entry
        if name not in METHOD_CONFIG:
            print(f"Unknown method '{name}' (expected one of {list(METHOD_CONFIG)}). Skipping.")
            continue
        parsed.append(parse_entry(entry))

    # Count runs per method so several values of the same one get distinct shades.
    per_method_total = defaultdict(int)
    for method, _ in parsed:
        per_method_total[method] += 1
    per_method_seen = defaultdict(int)

    curves = []          # (label, directory, filename, color, linestyle)
    reference_dir = None
    for method, value in parsed:
        d = method_dir(method, value)
        if not os.path.isdir(d):
            print(f"Warning: folder for '{method}' (value={value}) not found: {d}. Skipping.")
            continue
        cfg = METHOD_CONFIG[method]
        i = per_method_seen[method]
        per_method_seen[method] += 1
        color = shade(cfg['color'], i, per_method_total[method])
        curves.append((cfg['label'](value), d, 'sample_arr_run_0_it0.npy', color, cfg['linestyle']))
        if reference_dir is None:
            reference_dir = d

    if not curves:
        print("No valid method folders found -- nothing to plot.")
        return

    curves.append((REFERENCE_STYLE['label'], reference_dir, 'reference_arr.npy',
                   REFERENCE_STYLE['color'], REFERENCE_STYLE['linestyle']))

    for label, directory, file_name, color, linestyle in curves:
        batch_pattern = os.path.join(directory, "sample_batch*")
        try:
            fields = collect_and_average_data(batch_pattern, file_name)
        except FileNotFoundError as e:
            print(e)
            continue

        #  (a) Kinetic Energy Spectrum E(k)
        total_E_k = None
        k = None
        for field in fields:
            k, E_k = compute_ke_spectrum(field)
            total_E_k = E_k if total_E_k is None else total_E_k + E_k
        avg_E_k = total_E_k / len(fields)
        ax1.loglog(k, avg_E_k, color=color, linestyle=linestyle, linewidth=1.5, label=label)

        #  (b) Vorticity Distribution p(w) 
        all_vorticity = np.concatenate([f.flatten() for f in fields])
        kde = gaussian_kde(all_vorticity, bw_method=0.1)
        w_range = np.linspace(-10, 10, 500)
        ax2.plot(w_range, kde(w_range), color=color, linestyle=linestyle, linewidth=2, label=label)


    ax1.set_xlabel(r'$k$', fontsize=14)
    ax1.set_ylabel(r'$E(k)$', fontsize=14)
    ax1.set_xlim(left=1)
    ax1.grid(True, which="major", linestyle='-.', color='gray', alpha=0.5)
    ax1.set_title('(a) Kinetic energy spectrum', y=-0.2)
    ax1.legend(loc='lower left', framealpha=0.9)

    # --- Formatting Ax2: Vorticity Distribution ---
    ax2.set_xlabel(r'$\boldsymbol{\omega}$', fontsize=14)
    ax2.set_ylabel(r'$p(\boldsymbol{\omega})$', fontsize=14)
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(0, 0.12)
    ax2.grid(True, which="major", linestyle='-.', color='gray', alpha=0.5)
    ax2.set_title('(b) Vorticity distribution', y=-0.2)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Save and display
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    tag = "_".join(m if v is None else f"{m}{v}" for m, v in parsed)
    save_target = os.path.join("experiments", EXPERIMENT_FOLDER, f"stats_{tag}_vs_reference.png")
    plt.savefig(save_target, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {save_target}")
    plt.show()


if __name__ == "__main__":
    # Choose what to graph. Reference is ALWAYS included.
    #
    # Each entry is either
    #   "method"              -> uses that method's default value, or
    #   ("method", value)     -> that specific run.
    #
    #   "baseline"                  physics-guided diffusion   (no value)
    #   ("dps", 3.0)                posterior sampling         value = zeta
    #   "si"                        stochastic interpolant     (no value)
    #   ("si_linear", 0.01)         SI + linear physics        value = lambda
    #   ("si_learned", 3.0)         SI + learned physics       value = w
    #
    # The same method may appear several times at different values; repeats are
    # drawn in progressively lighter shades of that method's colour.
    #
    # e.g. lambda sweep:  [("si_linear", 0.001), ("si_linear", 0.01), ("si_linear", 0.1)]
    #      does physics help SI?  ["si", ("si_linear", 0.01)]
    #      three-method compare:  ["baseline", ("dps", 3.0), "si"]
    METHODS_TO_PLOT = ["baseline", "si", ("si_linear", 0.01)]

    plot_fluid_statistics(METHODS_TO_PLOT)