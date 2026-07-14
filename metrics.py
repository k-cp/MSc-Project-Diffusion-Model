import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
    Finds all matching .npy files across your batch folders and collects them.
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

# ---------------------------------------------------------------------------
# Experiment settings. Baseline and DPS save to different folders (the DPS one
# is prefixed with "dps_"), so we build each method's path from these pieces.
# ---------------------------------------------------------------------------
EXPERIMENT_FOLDER = "kmflow_re1000_rs256_ddim_conditional_new"
DATA_KW = "u3232"
T = 400
R = 20
W = 0.0
ZETA = 3.0   # which DPS zeta run to load (float, matches the --zeta you ran, e.g. 0.1, 3.0, 10.0)

# Style / folder-prefix for each method. Reference is handled separately below.
METHOD_CONFIG = {
    "baseline": {"prefix": "",     "label": "Baseline (physics-guided)", "color": "red",   "linestyle": "-"},
    "dps":      {"prefix": "dps_", "label": "Posterior sampling (DPS)",   "color": "green", "linestyle": "-"},
}
REFERENCE_STYLE = {"label": "Reference", "color": "mediumblue", "linestyle": "--"}


def method_dir(method):
    """Build the experiment folder path for a given method ('baseline' or 'dps').
    DPS runs live in a per-zeta folder (…_z<ZETA>)."""
    cfg = METHOD_CONFIG[method]
    folder = f"{cfg['prefix']}guided_recons_{DATA_KW}_t{T}_r{R}_w{W}"
    if method == "dps":
        folder += f"_z{ZETA}"
    return os.path.join("experiments", EXPERIMENT_FOLDER, folder)


def plot_fluid_statistics(methods_to_plot):
    """Plot E(k) and p(w). Always includes Reference, plus each selected method
    ('baseline' and/or 'dps')."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})

    # Build the list of curves: each selected method's reconstruction, then the
    # reference once (loaded from the first available method folder -- the
    # ground truth is identical across folders).
    curves = []          # (label, directory, filename, color, linestyle)
    reference_dir = None
    for method in methods_to_plot:
        if method not in METHOD_CONFIG:
            print(f"Unknown method '{method}' (expected 'baseline' or 'dps'). Skipping.")
            continue
        d = method_dir(method)
        if not os.path.isdir(d):
            print(f"Warning: folder for '{method}' not found: {d}. Skipping.")
            continue
        cfg = METHOD_CONFIG[method]
        curves.append((cfg['label'], d, 'sample_arr_run_0_it0.npy', cfg['color'], cfg['linestyle']))
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

        # ---- (a) Kinetic Energy Spectrum E(k) ----
        total_E_k = None
        k = None
        for field in fields:
            k, E_k = compute_ke_spectrum(field)
            total_E_k = E_k if total_E_k is None else total_E_k + E_k
        avg_E_k = total_E_k / len(fields)
        ax1.loglog(k, avg_E_k, color=color, linestyle=linestyle, linewidth=1.5, label=label)

        # ---- (b) Vorticity Distribution p(w) ----
        all_vorticity = np.concatenate([f.flatten() for f in fields])
        kde = gaussian_kde(all_vorticity, bw_method=0.1)
        w_range = np.linspace(-10, 10, 500)
        ax2.plot(w_range, kde(w_range), color=color, linestyle=linestyle, linewidth=2, label=label)

    # --- Formatting Ax1: Kinetic Energy Spectrum ---
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
    tag = "_".join(methods_to_plot)
    save_target = os.path.join("experiments", EXPERIMENT_FOLDER, f"stats_{tag}_vs_reference.png")
    plt.savefig(save_target, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {save_target}")
    plt.show()


if __name__ == "__main__":
    # Choose what to graph. Reference is ALWAYS included. Options:
    #   ["baseline"]          -> baseline vs reference
    #   ["dps"]               -> DPS vs reference
    #   ["baseline", "dps"]   -> both vs reference (three curves)
    METHODS_TO_PLOT = ["baseline", "dps"]

    plot_fluid_statistics(METHODS_TO_PLOT)