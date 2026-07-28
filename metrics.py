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
    # Same architecture/config as "baseline", but the weights we trained ourselves
    # instead of the ones shipped with the repo -- the reproduction check.
    "baseline_mine": {"method": "baseline", "variant": "mine"},
    # same, but trained WITH physics-valid x-translation augmentation
    "baseline_mine_xshift": {"method": "baseline", "variant": "mine_xshift"},
    # Physics ablation: which half of Shu et al.'s guidance does the work.
    "baseline_cond":   {"method": "baseline", "phys": "cond"},
    "baseline_linear": {"method": "baseline", "phys": "linear"},
    "baseline_nophys": {"method": "baseline", "phys": "none"},
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
    spec.setdefault("phys", "both")        # baseline physics ablation
    # Sampler-resolution knobs, so a t- or r-sweep can be compared on one plot:
    #   {"method":"baseline", "r":100}   or   {"method":"baseline", "t":300}
    spec.setdefault("t", T)
    spec.setdefault("r", R)
    spec.setdefault("w", W)                # classifier-free guidance weight
    spec.setdefault("ss", 1)               # JCP iterative-refinement rounds
    spec.setdefault("dps_cond", False)     # DPS physics CONDITIONING
    spec.setdefault("dps_phys", "none")    # DPS physics FORCE
    spec.setdefault("dps_lambda", 1.0)
    spec.setdefault("mn", 0.0)             # inference-time sensor noise
    # which experiment folder to read from -- cross-Re runs write to their own
    spec.setdefault("exp", EXPERIMENT_FOLDER)
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
    base = f"guided_recons_{DATA_KW}_t{spec['t']}_r{spec['r']}_w{spec['w']}"
    m = spec["method"]
    if m == "baseline":
        name = base
        # physics ablation tag (main.py --baseline_physics); 'both' stays untagged
        if spec["phys"] != "both":
            name += "_phys" + spec["phys"]
        # iterative-refinement rounds (main.py --sample_step); 1 stays untagged.
        # ORDER MATTERS: must match main.py, which appends _phys then _ss then _tag.
        if spec["ss"] != 1:
            name += "_ss{}".format(spec["ss"])
        # self-trained weights land in a _mine / _mine_xshift folder (--baseline_tag)
        if spec["variant"] in ("mine", "mine_xshift"):
            name += "_" + spec["variant"]
    elif m == "dps":
        name = f"dps_{base}_z{spec['value']}"
        # DPS+physics hybrid (main.py --dps_cond/--dps_physics/--dps_lambda)
        if spec["dps_cond"]:
            name += "_cond"
        if spec["dps_phys"] != "none":
            name += "_phys{}_lam{}".format(spec["dps_phys"], spec["dps_lambda"])
        if spec["variant"] == "mine":
            name += "_mine"
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
    # sensor noise is tagged last for EVERY method, matching main.py
    if spec["mn"]:
        name += "_mn{}".format(spec["mn"])
    return os.path.join("experiments", spec["exp"], name)


def spec_to_label(spec):
    """Human-readable curve name. Sensor noise is appended for EVERY method --
    without it a noise sweep produces several identically-labelled curves."""
    if spec.get("label"):
        return spec["label"]
    lbl = _base_label(spec)
    if spec.get("mn"):
        lbl += f"  [sensor noise {spec['mn']}]"
    return lbl


def _base_label(spec):
    m = spec["method"]
    if m == "baseline":
        phys = {"both": "", "cond": " · cond only", "linear": " · linear only",
                "none": " · no physics"}[spec["phys"]]
        # only mention t/r when they differ from the defaults, so ordinary
        # baseline labels stay short
        res = ""
        if spec["t"] != T:
            res += f" · t={spec['t']}"
        if spec["r"] != R:
            res += f" · r={spec['r']}"
        if spec["w"] != W:
            res += f" · w={spec['w']}"
        if spec["ss"] != 1:
            res += f" · ss={spec['ss']}"
        if spec["variant"] == "mine":
            return "Baseline (retrained here)" + phys + res
        if spec["variant"] == "mine_xshift":
            return "Baseline (retrained + x-shift)" + phys + res
        return "Baseline (provided weights)" + phys + res
    if m == "dps":
        lbl = f"DPS (zeta={spec['value']})"
        if spec["dps_cond"]:
            lbl += " + cond"
        if spec["dps_phys"] != "none":
            lbl += f" + force[{spec['dps_phys']}] λ={spec['dps_lambda']}"
        return lbl + " · retrained" if spec["variant"] == "mine" else lbl
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
        curves.append((spec_to_label(spec), d, spec_to_sample_file(spec), color, "-"))
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


def spec_to_sample_file(spec):
    """Which sample array holds the FINAL reconstruction for this run.

    A --sample_step N run writes one array per refinement round: it0 .. it{N-1}.
    it0 is the FIRST pass, NOT the answer -- reading it would compare the
    un-refined output and make JCP's iterative refinement look like a no-op.
    """
    return "sample_arr_run_0_it{}.npy".format(int(spec.get("ss", 1)) - 1)


def _load_full(folder, file_name):
    """Load ALL frames of a run (not just the last per batch)."""
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", file_name)))
    if not files:
        return None
    return np.concatenate([np.load(f) for f in files]).astype(np.float64)


def _resolve(methods_to_plot):
    """methods_to_plot -> [(label, folder, color, sample_file)]; reference from the first.

    sample_file is per-run because a sample_step>1 run stores its final result
    in it{N-1}, not it0 (see spec_to_sample_file).
    """
    out, ref_dir = [], None
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
        out.append((spec_to_label(spec), d, spec.get("color") or PALETTE[idx % len(PALETTE)],
                    spec_to_sample_file(spec)))
        if ref_dir is None:
            ref_dir = d
    return out, ref_dir


def metrics_table(methods_to_plot):
    """Per-run scalar metrics vs ground truth -- these EXPOSE the differences that
    the E(k)/p(w) plots hide (a genuinely different model shows a different RMSE
    even when its spectrum overlaps)."""
    runs, ref_dir = _resolve(methods_to_plot)
    if not runs:
        print("No valid folders -- nothing to tabulate.")
        return
    ref = _load_full(ref_dir, "reference_arr.npy")
    rstd = ref.std()

    print(f"\n{'method':30s} {'MSE':>9s} {'RMSE':>8s} {'corr':>8s} {'std':>8s} {'std/ref%':>9s}")
    print("-" * 76)
    print(f"{'reference (ground truth)':30s} {'--':>9s} {'--':>8s} {'--':>8s} {rstd:8.3f} {'100.0':>9s}")
    lines = [f"reference std={rstd:.4f}"]
    for label, d, _, sample_file in runs:
        x = _load_full(d, sample_file)
        if x is None:
            continue
        mse = float(((x - ref) ** 2).mean())
        rmse = float(np.sqrt(mse))
        corr = float(np.corrcoef(x.ravel(), ref.ravel())[0, 1])
        std = float(x.std())
        row = f"{label:30s} {mse:9.4f} {rmse:8.4f} {corr:8.4f} {std:8.3f} {100*std/rstd:9.1f}"
        print(row)
        lines.append(row)
    print("\n(lower MSE/RMSE = closer to truth; corr near 1 = better; std/ref near 100% = "
          "reproduces variance, <100% = over-smoothed.)")
    out = os.path.join("experiments", EXPERIMENT_FOLDER, "metrics_table.txt")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def plot_spectrum_ratio(methods_to_plot):
    """E(k) / E_reference(k) on a LINEAR y-axis. The raw log-log E(k) hides errors
    because it spans ~9 decades; this ratio makes the high-k over/under-shoot
    (where reconstructions actually differ) obvious. 1.0 = matches reference."""
    runs, ref_dir = _resolve(methods_to_plot)
    if not runs:
        print("No valid folders -- nothing to plot.")
        return

    def mean_ek(folder, fn):
        fields = collect_and_average_data(os.path.join(folder, "sample_batch*"), fn)
        total, k = None, None
        for field in fields:
            k, ek = compute_ke_spectrum(field)
            total = ek if total is None else total + ek
        return k, total / len(fields)

    k, e_ref = mean_ek(ref_dir, "reference_arr.npy")
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, d, color, sample_file in runs:
        _, e = mean_ek(d, sample_file)
        ax.semilogx(k, e / e_ref, color=color, linewidth=2, label=label)
    ax.axhline(1.0, color=REFERENCE_STYLE['color'], linestyle='--', linewidth=1.5,
               label="Reference (=1)")
    ax.set_xlabel(r'$k$', fontsize=14)
    ax.set_ylabel(r'$E(k)\,/\,E_{\mathrm{ref}}(k)$', fontsize=14)
    ax.set_ylim(0, 2.5)          # clip: values >2.5 (badly over-energetic) run off-top
    ax.set_xlim(left=1)
    ax.grid(True, which="major", linestyle='-.', color='gray', alpha=0.5)
    ax.set_title("Energy-spectrum ratio to reference "
                 "(1.0 = perfect; deviations reveal the hidden differences)")
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    save = os.path.join("experiments", EXPERIMENT_FOLDER,
                        f"spectrum_ratio_{len(runs)}runs.png")
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save}")
    plt.show()


def _mean_ek(folder, fn):
    """Mean energy spectrum over the per-batch representative fields (the same
    64-field sample the spectrum plots use -- one field per sample_batch dir)."""
    fields = collect_and_average_data(os.path.join(folder, "sample_batch*"), fn)
    total, k = None, None
    for field in fields:
        k, ek = compute_ke_spectrum(field)
        total = ek if total is None else total + ek
    return k, total / len(fields)


# Inertial-range band for the spectral-slope fit: the forcing sits at k=4 and the
# dissipation tail dominates above ~k=80, so fit the log-log slope between them.
SPECTRAL_FIT_BAND = (10, 60)


def deployability_metrics(methods_to_plot):
    """Standardised spectral scalars + derived quantities (Ren et al., Front. Mech.
    Eng. 2026 -- the 'deployability' checklist).

      e_spec  = sum|E_hat(k) - E_ref(k)| / sum E_ref(k)   integrated relative
                spectral error over the full resolved band (one scalar instead of
                our ad-hoc k-band percentages; lower = better).
      Delta_s = |s_hat - s_ref|, the log-log spectral-slope deviation fitted over
                SPECTRAL_FIT_BAND. The slope IS the cascade physics: a method can
                have plausible energy but the wrong kind of turbulence.
      KE%     = total kinetic energy vs reference (sum of the spectrum).
      Z%      = enstrophy <w^2> vs reference, computed on ALL 1272 frames. In 2D,
                energy dissipation = nu*<w^2>, so this is also the dissipation-
                rate ratio. Enstrophy weights exactly the high-k content where the
                methods differ most (DPS injects it, the baseline smooths it away).

    Spectra use the same 64-field per-batch sample as the spectrum plots;
    enstrophy uses the full test set (cheap, exact).
    """
    runs, ref_dir = _resolve(methods_to_plot)
    if not runs:
        print("No valid folders -- nothing to compute.")
        return
    k, e_ref = _mean_ek(ref_dir, "reference_arr.npy")
    band = (k >= SPECTRAL_FIT_BAND[0]) & (k <= SPECTRAL_FIT_BAND[1])
    logk = np.log(k[band])
    s_ref = float(np.polyfit(logk, np.log(e_ref[band]), 1)[0])
    ref_full = _load_full(ref_dir, "reference_arr.npy")
    z_ref = float((ref_full ** 2).mean())
    ke_ref = float(e_ref.sum())

    print(f"\nDeployability metrics (Ren et al. 2026): slope fit over k="
          f"[{SPECTRAL_FIT_BAND[0]},{SPECTRAL_FIT_BAND[1]}], ref slope s={s_ref:.2f}")
    print(f"{'method':30s} {'e_spec':>8s} {'slope':>7s} {'Δs':>6s} {'KE%':>7s} {'Z%':>7s}")
    print("-" * 70)
    print(f"{'reference (ground truth)':30s} {'0.000':>8s} {s_ref:7.2f} {'0.00':>6s} "
          f"{'100.0':>7s} {'100.0':>7s}")
    for label, d, _, sample_file in runs:
        try:
            _, e = _mean_ek(d, sample_file)
        except FileNotFoundError:
            continue
        e_spec = float(np.abs(e - e_ref).sum() / e_ref.sum())
        s_hat = float(np.polyfit(logk, np.log(np.maximum(e[band], 1e-30)), 1)[0])
        x = _load_full(d, sample_file)
        z = float((x ** 2).mean()) if x is not None else float("nan")
        print(f"{label:30s} {e_spec:8.3f} {s_hat:7.2f} {abs(s_hat - s_ref):6.2f} "
              f"{100 * e.sum() / ke_ref:7.1f} {100 * z / z_ref:7.1f}")
    print("\n(e_spec: integrated relative spectral error, lower better. Δs: cascade-"
          "slope deviation,\n 0 = right kind of turbulence. KE/Z: energy and "
          "enstrophy vs reference, 100 = perfect;\n Z<100 = smoothed, Z>100 = "
          "spurious high-k content. Dissipation ratio == Z ratio in 2D.)")


if __name__ == "__main__":
    # Reference is ALWAYS included. Every entry is one of:
    #   "baseline" / "dps" / "si" / "si_blind" / "si_linear" / "si_learned"   (shorthands)
    #   ("dps", 3.0) / ("si_linear", 0.01)                                    (shorthand + value)
    #   {"method": "si", "eval": "sensor:512"}
    #   {"method":"si", "variant":"blind", "eval":"sensor:512", ...}          (full control)
    #
    # PHYSICS ABLATION -- which half of Shu et al.'s guidance actually does the work.
    # "baseline" is conditioning + direct descent (the repo default, both mechanisms);
    # "baseline_cond" keeps the conditioning but drops the '- dx' subtraction --
    # that pair is the CLEAN comparison, since both use the well-trained conditional
    # branch and differ only in the sampler.
    # "baseline_nophys"/"baseline_linear" fall back to the UNCONDITIONAL branch, which
    # saw only p=0.1 of training samples, so read those two as indicative, not decisive.
    # REPRODUCTION CHECK: the checkpoint shipped with the repo vs one trained here
    # from scratch. Same architecture, same config, same 36-trajectory train split,
    # same sampler -- so any gap is down to the provided checkpoint's unknown
    # training. If the two agree, every result resting on "baseline" is sound; if
    # they diverge, the retrained one is the honest reference to quote.
    # KEEP "baseline" FIRST: the reference arrays are read from the first resolved
    # folder, and that is the one guaranteed to be present locally.
    # HEADLINE COMPARISON -- the best of everything that has actually been run.
    # r=100 is included because the default baseline gets only 20 reverse steps
    # against SI's 100; at matched resolution its NS residual drops 62.8 -> 8.85
    # (SI: 8.28), so quoting only the r=20 baseline overstates SI's PHYSICS
    # advantage. Its FIELD-ACCURACY advantage (1.45 vs ~3.6) survives either way.
    METHODS_TO_PLOT = [
        "baseline",                                  # exact sensors (as published)
        {"method": "baseline", "mn": 0.05},          # 5% sensor error
        ("dps", 3.0),
        {"method": "dps", "value": 3.0, "mn": 0.05},
        "si",
        {"method": "si", "mn": 0.05},
    ]

    # For the full 3x3 grid with degradation ratios, use the dedicated script:
    #     python noise_robustness.py
    #
    # Other sets worth swapping in (ALL of these now exist locally):
    #   noise robustness  ["baseline",{"method":"baseline","mn":0.02},{"method":"baseline","mn":0.05}]
    #   headline          ["baseline",{"method":"baseline","r":100},"baseline_nophys",
    #                      ("dps",3.0),{"method":"dps","value":3.0,"dps_phys":"x0hat","dps_lambda":1.0},"si"]
    #   physics ablation  ["baseline","baseline_cond","baseline_linear","baseline_nophys"]
    #     -> rows pair off by the '- dx' SUBTRACTION; the CONDITIONING is inert (0.03%).
    #        subtraction costs ~3% MSE and 3pp variance, buys a 6.7x lower residual.
    #   reproduction      ["baseline","baseline_mine"]   (needs the training job to finish)
    #   reverse steps     ["baseline",{"method":"baseline","r":50},{"method":"baseline","r":100}]
    #   JCP iterations    ["baseline",{"method":"baseline","ss":3},{"method":"baseline","ss":5}]
    #   guidance weight   ["baseline",{"method":"baseline","w":3.0},{"method":"baseline","w":3.0,"ss":3}]
    #   noise level       ["baseline",{"method":"baseline","t":300},{"method":"baseline","t":200}]
    #   DPS + physics     ["baseline",("dps",3.0),
    #                      {"method":"dps","value":3.0,"dps_cond":True},
    #                      {"method":"dps","value":3.0,"dps_phys":"x0hat","dps_lambda":1.0},
    #                      {"method":"dps","value":3.0,"dps_cond":True,"dps_phys":"x0hat","dps_lambda":1.0}]
    #   method headline   ["baseline",("dps",3.0),"si"]
    #   SI robustness     ["si","si_blind",{"method":"si","variant":"blind","eval":"sensor:512"}]

    # --- which outputs to produce (selectable) ---
    SHOW_SPECTRUM_DISTRIBUTION = True   # (a) E(k) log-log + (b) p(w) -- the paper's plots
    SHOW_SPECTRUM_RATIO        = True   # E(k)/E_ref, linear axis -- EXPOSES hidden high-k differences
    PRINT_METRICS_TABLE        = True   # numeric RMSE / corr / std -- exposes pointwise differences
    PRINT_DEPLOYABILITY        = True   # e_spec / slope deviation / KE / enstrophy (Ren et al. 2026)

    if SHOW_SPECTRUM_DISTRIBUTION:
        plot_fluid_statistics(METHODS_TO_PLOT)
    if PRINT_METRICS_TABLE:
        metrics_table(METHODS_TO_PLOT)
    if SHOW_SPECTRUM_RATIO:
        plot_spectrum_ratio(METHODS_TO_PLOT)
    if PRINT_DEPLOYABILITY:
        deployability_metrics(METHODS_TO_PLOT)