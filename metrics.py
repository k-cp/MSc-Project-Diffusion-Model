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
# Defaults used when an entry gives no explicit value.
ZETA = 3.0        # DPS      --zeta
SI_LAMBDA = 0.01  # SI linear   --si_lambda
SI_W = 3.0        # SI learned  --si_w

REFERENCE_STYLE = {"label": "Reference", "color": "mediumblue", "linestyle": "--"}

# Distinct colours assigned in order to curves that don't set an explicit one.
PALETTE = ["red", "green", "darkorange", "teal", "purple", "saddlebrown",
           "royalblue", "magenta", "olive", "darkviolet"]

# Shorthand string -> canonical spec. Anything else must be a dict spec.
_SHORTHAND = {
    "baseline":   {"method": "baseline"},
    "dps":        {"method": "dps"},
    "si":         {"method": "si"},
    "si_blind":   {"method": "si", "variant": "blind"},
    "si_linear":  {"method": "si", "physics": "linear"},
    "si_learned": {"method": "si", "physics": "learned"},
}


def normalize(entry):
    """Turn any entry into a canonical spec dict. Accepted forms:

        "si"                                         shorthand
        ("dps", 3.0) / ("si_linear", 0.1)            shorthand + value
        {"method":"si", "variant":"blind",           full control -- ANY combo
         "physics":"linear", "value":0.01,
         "eval":"sensor:512", "label":..., "color":...}

    Spec keys: method (baseline|dps|si); value (dps zeta / si physics strength);
    physics (none|linear|learned); variant (plain|blind); eval (e.g. sensor:512);
    label, color (optional overrides).
    """
    value = None
    if isinstance(entry, (tuple, list)):
        entry, value = entry[0], entry[1]
    if isinstance(entry, str):
        if entry not in _SHORTHAND:
            raise ValueError(f"unknown shorthand '{entry}'; use a dict spec for custom runs")
        spec = dict(_SHORTHAND[entry])
    else:
        spec = dict(entry)                       # already a dict spec

    if value is not None:
        spec["value"] = value
    spec.setdefault("method", "si")
    spec.setdefault("physics", "none")
    spec.setdefault("variant", "plain")
    spec.setdefault("eval", None)
    if "value" not in spec:
        if spec["method"] == "dps":
            spec["value"] = ZETA
        elif spec["physics"] == "linear":
            spec["value"] = SI_LAMBDA
        elif spec["physics"] == "learned":
            spec["value"] = SI_W
        else:
            spec["value"] = None
    return spec


def spec_to_folder(spec):
    """Build the experiment folder for a spec -- MUST match main.py's naming."""
    base = f"guided_recons_{DATA_KW}_t{T}_r{R}_w{W}"
    m = spec["method"]
    if m == "baseline":
        name = base
    elif m == "dps":
        name = f"dps_{base}_z{spec['value']}"
    elif m == "si":
        name = f"si_{base}"
        if spec["physics"] == "linear":
            name += f"_linear_lam{spec['value']}"
        elif spec["physics"] == "learned":
            name += f"_learned_w{spec['value']}"
        if spec["variant"] == "blind":
            name += "_blind"
        if spec["eval"]:
            name += "_eval_" + str(spec["eval"]).replace(":", "")
    else:
        raise ValueError(f"unknown method {m!r}")
    return os.path.join("experiments", EXPERIMENT_FOLDER, name)


def spec_to_label(spec):
    if spec.get("label"):
        return spec["label"]
    m = spec["method"]
    if m == "baseline":
        return "Baseline (physics-guided)"
    if m == "dps":
        return f"DPS (zeta={spec['value']})"
    parts = ["SI"]
    if spec["variant"] == "blind":
        parts.append("blind")
    if spec["physics"] == "linear":
        parts.append(f"linear λ={spec['value']}")
    elif spec["physics"] == "learned":
        parts.append(f"learned w={spec['value']}")
    if spec["eval"]:
        parts.append(f"@{spec['eval']}")
    return "Stochastic interpolant" if len(parts) == 1 else " · ".join(parts)


# Backward-compat shims (energy.py and older callers import these).
def parse_entry(entry):
    spec = normalize(entry)
    return spec["method"], spec["value"]


def method_dir(method, value=None):
    return spec_to_folder(normalize((method, value) if value is not None else method))


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

    # Resolve every entry (shorthand string, tuple, or full dict spec) into a
    # canonical spec, then into its folder + label + colour.
    curves = []          # (label, directory, filename, color, linestyle)
    reference_dir = None
    for idx, entry in enumerate(methods_to_plot):
        try:
            spec = normalize(entry)
        except ValueError as e:
            print(f"Skipping {entry!r}: {e}")
            continue
        d = spec_to_folder(spec)
        if not os.path.isdir(d):
            print(f"Warning: folder not found, skipping: {d}")
            continue
        color = spec.get("color") or PALETTE[idx % len(PALETTE)]
        curves.append((spec_to_label(spec), d, 'sample_arr_run_0_it0.npy', color, "-"))
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
    n = len(curves) - 1                          # exclude the reference curve
    save_target = os.path.join("experiments", EXPERIMENT_FOLDER,
                               f"stats_{n}runs_vs_reference.png")
    plt.savefig(save_target, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {save_target}")
    plt.show()


if __name__ == "__main__":
    # Choose what to graph. Reference is ALWAYS included. Every entry is one of:
    #
    #   "baseline" / "dps" / "si" / "si_blind" / "si_linear" / "si_learned"   (shorthands)
    #   ("dps", 3.0) / ("si_linear", 0.01)                                    (shorthand + value)
    #   {"method":"si", ...}                                                  (full control -- ANY combo)
    #
    # Dict spec keys: method, physics (none|linear|learned), value (zeta/lambda/w),
    #                 variant (plain|blind), eval (e.g. "sensor:512"), label, color.
    #
    # Examples:
    #   three-method compare:   ["baseline", ("dps", 3.0), "si"]
    #   specialist vs blind:    ["si", "si_blind"]
    #   robustness at 512 sensors (specialist vs blind):
    #       [{"method":"si", "eval":"sensor:512"},
    #        {"method":"si", "variant":"blind", "eval":"sensor:512"}]
    #   blind + linear physics:
    #       [{"method":"si", "variant":"blind", "physics":"linear", "value":0.01}]
    METHODS_TO_PLOT = ["baseline", "si", ("si_linear", 0.01)]

    plot_fluid_statistics(METHODS_TO_PLOT)