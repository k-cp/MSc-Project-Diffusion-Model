"""Look at the actual fields: truth, masked input, each model's fill, and its error.

    python plot_si_fields.py --geom single --gap 0.6              # SI vs FNO, one frame
    python plot_si_fields.py --geom single --gap 0.6 --frame 640
    python plot_si_fields.py --geom multiple --gap 25 --models SI,FNO

Runs ON THE CLUSTER, where the arrays are. Reads the same run folders eval_si_inpaint
scores, resolved by the same folder() function, so what you look at is what was
measured -- not a separate re-derivation that could drift from the table.

THE ZOOM PANEL IS THE POINT. A 0.6% void is 20x20 pixels in a 256x256 field: at full
size it is a speck, and a figure of the whole field shows nothing but that the model
copied the other 99.4% correctly. Every row therefore gets a crop around the void's
bounding box with a margin, which is where the actual reconstruction lives.

COLOUR SCALES ARE SHARED AND SYMMETRIC ACROSS A ROW. Vorticity is signed and roughly
symmetric, so an independent scale per panel would make a washed-out fill look like a
correct one; the error panels share their own scale across models, so two models can be
compared by eye rather than by reading three different colourbars.

WHAT THIS CANNOT SHOW. One frame is one draw. The metric table is 1272 frames, and a
figure chosen from among them is an illustration, not evidence -- pick the frame index
before looking at the results, or say that you did not.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_si_inpaint import (folder, MODEL_VARIANTS, MODEL_LABEL,   # noqa: E402
                             AUX_VARIANTS, AUX_LABEL, RECIPE_VARIANTS,
                             RECIPE_LABEL, CORR_LEN)

DEFAULT_ROOT = ("/dss/dssfs04/lwp-dss-0002/pn46yo/pn46yo-dss-0000/di24lir/"
                "experiments/kf_2d_re1000_si_inpaint")


def load_frame(run_dir, frame):
    """(truth, prediction) for one global frame index.

    BATCHES ARE SORTED NUMERICALLY HERE, not lexicographically as in the eval. The
    eval sorts lexicographically because sums do not care about order; a frame INDEX
    does, and sample_batch10 sorting after sample_batch1 would silently address a
    different snapshot than the one asked for.
    """
    batches = sorted(glob.glob(os.path.join(run_dir, "sample_batch*")),
                     key=lambda p: int(p.rsplit("batch", 1)[-1]))
    if not batches:
        return None, None
    seen = 0
    for b in batches:
        ref = np.load(os.path.join(b, "reference_arr.npy"), mmap_mode="r")
        if seen + ref.shape[0] > frame:
            i = frame - seen
            prd = np.load(os.path.join(b, "sample_arr_run_0_it0.npy"), mmap_mode="r")
            return np.asarray(ref[i], float), np.asarray(prd[i], float)
        seen += ref.shape[0]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--geom", default="single",
                    choices=("single", "multiple", "random"))
    ap.add_argument("--gap", type=float, default=0.6)
    ap.add_argument("--n_voids", type=int, default=None)
    ap.add_argument("--trained", default="random", choices=("random", "center"))
    ap.add_argument("--models", default=",".join(MODEL_LABEL[m] for m in MODEL_VARIANTS),
                    help="Comma-separated labels, e.g. SI,FNO")
    ap.add_argument("--aux", default="-",
                    help="Aux label(s), comma-separated: - / perc / gan / p+g / res. "
                         "'plain' is accepted for '-'. "
                         "Several makes one column per objective, which is the only way "
                         "to see what an adversarial or perceptual term actually did to "
                         "the FILL -- the metric table says variance went up, the "
                         "picture says whether that is structure or noise.")
    ap.add_argument("--recipe", default="ema+xshift",
                    choices=tuple(RECIPE_LABEL.values()))
    ap.add_argument("--frame", type=int, default=640,
                    help="Global frame index into the 1272-frame test split. Pick it "
                         "BEFORE looking at the metrics, or say that you did not.")
    ap.add_argument("--baselines", default="",
                    help="Classical fills to add as extra columns, comma-separated: "
                         "biharmonic, harmonic, zero. Computed ON THE FLY from the "
                         "truth and mask already in the run folder -- no second run to "
                         "schedule and no arrays to move, because a classical fill is a "
                         "function of the masked field alone. They are the SAME "
                         "functions baselines_inpaint.py scores, so the picture and the "
                         "floor in the table cannot disagree.")
    ap.add_argument("--margin", type=int, default=24,
                    help="Pixels of context around the void in the zoom row")
    ap.add_argument("--per-row", type=int, default=0, dest="per_row",
                    help="wrap the panels onto rows of this many. 0 keeps the "
                         "single-row-per-view layout, which is 2 + 2*models wide "
                         "and unreadable past three models. 4 is a good page fit.")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    inv_model = {v: k for k, v in MODEL_LABEL.items()}
    inv_aux = {v: k for k, v in AUX_LABEL.items()}
    # 'plain' as an alias for the '-' label: argparse reads a value starting with '-'
    # as a flag, so `--aux -,perc` fails and `--aux="-,perc"` is a trap nobody
    # remembers. The eval's table still prints '-'; this is a CLI convenience only.
    inv_aux["plain"] = ""
    inv_rec = {v: k for k, v in RECIPE_LABEL.items()}
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    want_aux = [x.strip() for x in a.aux.split(",") if x.strip()]

    # COLUMNS ARE (model, aux) PAIRS, not models. One model with five objectives and
    # five models with one objective are the same figure to everything downstream.
    found = []
    for label in want:
        if label not in inv_model:
            raise SystemExit(f"unknown model {label!r}; choose from "
                             f"{sorted(inv_model)}")
        for auxlab in want_aux:
            if auxlab not in inv_aux:
                raise SystemExit(f"unknown aux {auxlab!r}; choose from "
                                 f"{sorted(inv_aux)}")
            col = label if auxlab == "-" else (
                auxlab if len(want) == 1 else f"{label} {auxlab}")
            d, info = folder(a.root, a.geom, a.gap, a.trained, a.n_voids,
                             inv_aux[auxlab], inv_rec[a.recipe], inv_model[label])
            if not os.path.isdir(d):
                print(f"  SKIP {col}: no run at {os.path.basename(d)}")
                continue
            ref, prd = load_frame(d, a.frame)
            if ref is None:
                print(f"  SKIP {col}: frame {a.frame} beyond this run")
                continue
            mask = np.load(os.path.join(d, "void_mask.npy")).astype(bool)
            found.append((col, ref, prd, mask, info))
    if not found:
        raise SystemExit("no runs found -- check --root, --geom/--gap, --recipe")

    _, ref, _, mask, info = found[0]
    void = ~mask

    # --- classical fills, appended as further columns.
    # DETERMINISTIC GIVEN (truth, mask), which is why they can be produced here rather
    # than read from a run: there is no training and no checkpoint. Deliberately the
    # same implementations the metric table uses.
    for bl in [b.strip() for b in a.baselines.split(",") if b.strip()]:
        try:
            if bl == "biharmonic":
                from baselines_inpaint import fill_biharmonic
                pred = fill_biharmonic(ref[None], void)[0]
            elif bl == "harmonic":
                from baselines_inpaint import HarmonicFill
                pred = HarmonicFill(void)(ref[None])[0]
            elif bl == "zero":
                from baselines_inpaint import fill_zero
                pred = fill_zero(ref[None], void)[0]
            else:
                raise SystemExit(f"unknown baseline {bl!r}: "
                                 "choose from biharmonic, harmonic, zero")
        except ImportError as exc:
            # skimage is the only non-core dependency here and it may be absent on a
            # compute node. Say so and carry on rather than losing the whole figure.
            print(f"  SKIP {bl}: {exc}")
            continue
        found.append((bl, ref, np.asarray(pred, float), mask, info))

    ys, xs = np.where(void)
    y0, y1 = max(ys.min() - a.margin, 0), min(ys.max() + a.margin + 1, void.shape[0])
    x0, x1 = max(xs.min() - a.margin, 0), min(xs.max() + a.margin + 1, void.shape[1])
    blur = ref * mask                        # what every model was handed

    ncol = 2 + 2 * len(found)
    # Each view (full field, zoom) needs `ncol` panels. Wrapping them onto rows of
    # `per_row` keeps the panels legible once there are more than about three models.
    per_row = a.per_row if a.per_row > 0 else ncol
    rows_per_view = -(-ncol // per_row)                 # ceil
    nrow = 2 * rows_per_view
    fig, axes = plt.subplots(nrow, per_row,
                             figsize=(2.5 * per_row, 3.3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()                               # blanks any unused cell

    def slot(view, i):
        """Panel i of view (0 full field, 1 zoom) -> axis, wrapped onto rows."""
        ax = axes[view * rows_per_view + i // per_row][i % per_row]
        ax.set_axis_on()
        return ax

    v = float(np.abs(ref).max())
    errs = [np.abs(p - ref) for _, _, p, _, _ in found]
    emax = float(max(e[void].max() for e in errs)) if errs else 1.0

    def show(ax, img, title, cmap, vmin, vmax, crop=False):
        d = img[y0:y1, x0:x1] if crop else img
        h = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return h

    for row, crop in enumerate((False, True)):
        tag = "  [zoom]" if crop else ""
        show(slot(row, 0), ref, f"truth{tag}", "RdBu_r", -v, v, crop)
        show(slot(row, 1), blur, f"input (void = 0){tag}", "RdBu_r", -v, v, crop)
        for j, (label, _, prd, _, _) in enumerate(found):
            show(slot(row, 2 + 2 * j), prd, f"{label}{tag}", "RdBu_r", -v, v, crop)
            eax = slot(row, 3 + 2 * j)
            h = show(eax, np.abs(prd - ref),
                     f"|{label} - truth|{tag}", "magma", 0, emax, crop)
            if row == 1:
                plt.colorbar(h, ax=eax, fraction=0.046)
        # Outline the void in the full-field row so the reader can find it at all.
        if not crop:
            for ax in axes[row * rows_per_view: (row + 1) * rows_per_view].ravel():
                if not ax.has_data():
                    continue
                ax.contour(void.astype(float), levels=[0.5], colors="k",
                           linewidths=0.7)

    nv = info["n_voids"]
    fig.suptitle(
        f"{a.geom}" + ("" if a.geom == "single" else f" x{nv}") +
        f" {a.gap}%  --  void {info['max_depth']:.0f} px deep "
        f"({info['max_depth'] / CORR_LEN:.2f} correlation lengths), frame {a.frame}\n"
        f"trained on a {a.trained} void placement, scored on the fixed centred one; "
        f"error panels share one scale",
        fontsize=10)
    # h_pad: without it the zoom row's titles sit on top of the full-field row.
    fig.tight_layout(rect=[0, 0, 1, 0.9], h_pad=2.4)
    os.makedirs(a.out, exist_ok=True)
    nvtag = "" if a.geom == "single" else f"_n{nv}"
    # A centre-trained run and a re-rolled one are different models scored on the same
    # mask, and an aux comparison is a different figure again -- all three would
    # otherwise write the same filename.
    tags = ("" if a.trained == "random" else "_ctrtrain") + \
           ("" if len(want_aux) < 2 else "_auxcmp")
    p = os.path.join(a.out,
                     f"fields_{a.geom}{nvtag}_g{a.gap:g}{tags}_f{a.frame}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)

    # The numbers behind the picture, so a striking frame can be checked against the
    # run as a whole rather than taken at face value.
    print(f"\n  in-void relL2 for THIS frame (the table's number is over all 1272):")
    den = np.sqrt((ref[void] ** 2).sum())
    for label, _, prd, _, _ in found:
        e = np.sqrt(((prd - ref)[void] ** 2).sum()) / den
        print(f"    {label:>5}: {e:.4f}")


if __name__ == "__main__":
    main()
