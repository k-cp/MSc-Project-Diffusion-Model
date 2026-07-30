"""Figure: how much injected sensor noise SURVIVES into each method's output.

This is the picture behind the chapter's central claim. --meas_noise is applied
in STANDARDIZED units (rs256_guided_diffusion.py:190-204: x + sigma*randn over
the whole measurement field), and the runtime scaler's scale is the TRAINING
split's std (`data_scale = np.std(ref_data[:-4])`, :81) = 4.7869, the value in
train_ddpm/km256_stats.npz. So the injected noise variance in physical vorticity
units is (sigma * 4.7869)^2, and the share of it that reaches the output is

    pass-through = [ MSE(sigma) - MSE(0) ] / (sigma * scale)^2

A method that destroys the measurement before generating cannot pass the error
on; one that starts AT the measurement passes it through ~1:1. That single
mechanism is also why the first is less accurate -- the point of the figure is
that these are one decision, not two findings.

    python noise_passthrough_plot.py
"""

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import normalize, spec_to_folder, spec_to_sample_file

SIGMAS = [0.02, 0.05]
SCALE = 4.7869                      # train-split std; see module docstring
OUT = "noise_passthrough.png"

# Same hues as the experiment-inventory / progress pages, so the whole project
# reads as one set: blue baseline, violet DPS, rust SI.
METHODS = [
    ("Diffusion baseline", "#2f6ea8", lambda mn: {"method": "baseline", "mn": mn}),
    ("DPS (zeta=3.0)",     "#7a5296", lambda mn: {"method": "dps", "value": 3.0, "mn": mn}),
    ("Stochastic interp.", "#b04a2f", lambda mn: {"method": "si", "mn": mn}),
]


def stream_mse(folder, sample_file, ref_by_batch):
    """MSE against the reference, accumulated batch by batch.

    Streaming rather than _load_full: each run is ~670 MB as float64 and we walk
    nine of them, so holding one at a time keeps this comfortable on a laptop.
    """
    files = sorted(glob.glob(os.path.join(folder, "sample_batch*", sample_file)))
    if not files:
        return None
    tot, n = 0.0, 0
    for f in files:
        b = os.path.basename(os.path.dirname(f))
        ref = ref_by_batch.get(b)
        if ref is None:
            return None
        x = np.load(f).astype(np.float64)
        if x.shape != ref.shape:
            return None
        tot += float(((x - ref) ** 2).sum())
        n += x.size
    return tot / n


def reference_by_batch(folder):
    """reference_arr keyed by batch dir, so runs are compared frame-for-frame."""
    out = {}
    for f in sorted(glob.glob(os.path.join(folder, "sample_batch*", "reference_arr.npy"))):
        out[os.path.basename(os.path.dirname(f))] = np.load(f).astype(np.float64)
    return out


def main():
    ref_dir = spec_to_folder(normalize({"method": "baseline"}))
    ref = reference_by_batch(ref_dir)
    if not ref:
        print(f"No reference_arr under {ref_dir}; cannot proceed.")
        return
    print(f"Reference: {len(ref)} batches from {ref_dir}")

    rows = []
    for label, colour, fn in METHODS:
        spec0 = normalize(fn(0.0))
        mse0 = stream_mse(spec_to_folder(spec0), spec_to_sample_file(spec0), ref)
        if mse0 is None:
            print(f"  {label}: no noise-free run, skipped")
            continue
        entry = {"label": label, "colour": colour, "mse0": mse0, "pt": {}}
        for s in SIGMAS:
            spec = normalize(fn(s))
            mse = stream_mse(spec_to_folder(spec), spec_to_sample_file(spec), ref)
            if mse is None:
                print(f"  {label} sigma={s}: missing, skipped")
                continue
            entry["pt"][s] = (mse - mse0) / (s * SCALE) ** 2
            entry.setdefault("mse", {})[s] = mse
        rows.append(entry)
        print(f"  {label:22s} MSE(0)={mse0:7.4f}  " + "  ".join(
            f"sigma={s}: MSE={entry['mse'][s]:7.4f} pass-through={100*entry['pt'][s]:6.1f}%"
            for s in SIGMAS if s in entry["pt"]))

    if not rows:
        print("Nothing to plot.")
        return

    # --- figure -------------------------------------------------------------
    # Dots, not bars: the x-axis spans two decades, and a bar on a log axis has
    # a length that is not proportional to its value, which reads as a much
    # smaller difference than 2% vs 120% actually is.
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ypos = np.arange(len(rows), dtype=float)

    for y, r in zip(ypos, rows):
        vals = [100 * r["pt"][s] for s in SIGMAS if s in r["pt"]]
        if len(vals) > 1:                      # tie the same method's points together
            ax.plot([min(vals), max(vals)], [y, y], color=r["colour"],
                    lw=1.4, alpha=0.45, zorder=1, solid_capstyle="round")
        for s in SIGMAS:
            if s not in r["pt"]:
                continue
            v = 100 * r["pt"][s]
            filled = (s == SIGMAS[-1])
            ax.plot(v, y, marker="o", ms=9.5, zorder=3,
                    color=r["colour"] if filled else "white",
                    markeredgecolor=r["colour"], markeredgewidth=1.8)
        ax.text(max(vals) * 1.22, y, " / ".join(
                    f"{100*r['pt'][s]:.1f}%" for s in SIGMAS if s in r["pt"]),
                va="center", ha="left", fontsize=9.5, color="#141b24")

    ax.axvline(100, color="#8c96a3", lw=1.1, ls="--", zorder=0)
    ax.text(100, -0.72, "noise passes through 1:1 ", color="#5d6b7c",
            fontsize=9, va="center", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(1.2, 700)
    ax.set_ylim(len(rows) - 0.4, -0.95)         # inverted, with headroom for the note
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("share of injected sensor-noise variance reaching the output (%, log scale)")
    ax.set_title("Destroying the measurement buys noise immunity — and costs accuracy",
                 fontsize=12.5, loc="left", pad=14)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9.5, color="white",
                          markeredgecolor="#5d6b7c", markeredgewidth=1.8,
                          label=f"$\\sigma$ = {SIGMAS[0]}"),
               plt.Line2D([], [], marker="o", ls="", ms=9.5, color="#5d6b7c",
                          markeredgecolor="#5d6b7c", markeredgewidth=1.8,
                          label=f"$\\sigma$ = {SIGMAS[-1]}")]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right",
              ncol=2, columnspacing=1.4, handletextpad=0.4)
    ax.grid(axis="x", alpha=0.22, which="major")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=180)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
