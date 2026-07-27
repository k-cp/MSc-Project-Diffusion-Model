"""Plot the SI robustness sweep: error vs sensor count, specialist vs blind.

The headline figure for the robustness experiment. Reads the same eval-run logs
as si_robustness.py (reusing its resolver + parser) and draws L2 (and residual)
against sensor density on a log-x axis, one line per trained checkpoint.

Missing (variant, sensor) runs are skipped, so this is safe to run mid-sweep --
it draws whatever points exist and fills in as more jobs finish.

    python si_robustness_plot.py            # -> si_robustness_curve.png

The real u3232 input (1024 FIXED sensor positions, what the specialist trained on)
is drawn as a separate marker, distinct from random sensor:1024, because the two
are different degradations that happen to share a sensor count.
"""

import re
import matplotlib.pyplot as plt

from si_robustness import VARIANTS, eval_folder, read_metrics

# Random-sensor densities to place on the x-axis (parsed as "sensor:N").
SENSOR_COUNTS = [256, 512, 1024, 2048]

# Line style per trained checkpoint.
STYLE = {"specialist": dict(color="C0", marker="o"),
         "blind":      dict(color="C1", marker="s")}


def collect(metric_index):
    """metric_index: 0 = L2, 1 = residual. Returns {variant: (xs, ys)} for the
    random-sensor sweep, plus {variant: y} for the real u3232 (1024-fixed) point."""
    sweep, u3232 = {}, {}
    for v, tag in VARIANTS.items():
        xs, ys = [], []
        for n in SENSOR_COUNTS:
            m = read_metrics(eval_folder(tag, f"sensor:{n}"))[metric_index]
            if m is not None:
                xs.append(n)
                ys.append(m)
        sweep[v] = (xs, ys)
        u3232[v] = read_metrics(eval_folder(tag, ""))[metric_index]
    return sweep, u3232


def annotate_slope(ax, sweep, lo=256, hi=1024):
    """Print each variant's degradation FACTOR over hi->lo sensors.

    This is the headline of the robustness experiment: if blind augmentation had
    bought robustness, its factor would be SMALLER (a flatter curve). Measured,
    the two are ~equal -- the curves are parallel, so augmentation only shifted
    the line up. Drawn on the plot because two similar lines don't show it.
    """
    lines = []
    for v, (xs, ys) in sweep.items():
        pt = dict(zip(xs, ys))
        if lo in pt and hi in pt and pt[hi]:
            lines.append(f"{v}: {pt[lo]/pt[hi]:.2f}x")
    if len(lines) < 2:
        return
    ax.text(0.03, 0.97, f"degradation {hi}->{lo} sensors\n" + "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))


def draw(ax, metric_index, ylabel, title, slope=False):
    sweep, u3232 = collect(metric_index)
    for v in VARIANTS:
        xs, ys = sweep[v]
        if xs:
            ax.plot(xs, ys, label=f"{v} (random sensors)", **STYLE[v])
        y = u3232[v]
        if y is not None:
            ax.plot(1024, y, markerfacecolor="none", markeredgewidth=2,
                    linestyle="none", markersize=11,
                    label=f"{v} @ u3232 (1024 fixed)",
                    color=STYLE[v]["color"], marker=STYLE[v]["marker"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(SENSOR_COUNTS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("sensor count (random)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if slope:
        annotate_slope(ax, sweep)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw(axes[0], 0, "Mean L2 error (lower better)", "Field error vs sensor density",
         slope=True)
    draw(axes[1], 1, "Mean NS residual (lower better)", "Physics residual vs sensor density")
    fig.suptitle("SI robustness: specialist vs blind across sensor densities")
    fig.tight_layout()
    out = "si_robustness_curve.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
