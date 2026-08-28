# HANDOFF — end of 2026-08-28 session

Supersedes the 2026-08-27 version. Deadline: **September 2026** (~33 days).
Writing is the critical path. All training is finished; one inference job is
queued and everything else is writing.

---

## 1. DO THESE FIRST (in order)

1. **Check the U-Net inference.** Two identical jobs were submitted (5765515,
   5765839) and **both were PENDING** at session end. Cancel the duplicate if
   it has not already run — the inference runner does `shutil.rmtree(batch_dir)`
   before writing, so two concurrent copies would delete each other's output:
   ```
   scancel 5765839          # keep 5765515
   ```
   When it completes, verify it produced samples rather than just exiting 0:
   ```
   ls ~/Diffusion-based-Fluid-Super-resolution/experiments/*/*unet*/sample_batch*/sample_arr_run_0_it0.npy | wc -l
   ```
   **Expect 64** (1,272 frames / batch 20). Fewer means it died partway, which
   a clean exit code will not tell you. Then:
   ```
   python eval_si_inpaint.py --csv figures/metrics_si_inpaint.csv
   ```
   and pull that CSV to the Mac. `make_master_table.py` picks up the U-Net
   column automatically — `_unet` is already in `MODEL_VARIANTS`.

   Training itself is DONE and verified: checkpoint
   `fno_inpaint_single_g25_ctr_unet.pth` reports `epoch reached: 2000`,
   job 5763240, 26.6 h.

2. **Evaluate the four raw-transfer runs.** Still outstanding from the last
   handoff. They FINISHED (jobs 5763217–5763220, ~15 min each).
   ```
   for re in 500 1000 2000 10000; do
     python eval_si_inpaint.py \
       --root ~/Diffusion-based-Fluid-Super-resolution/experiments/kf_re${re}_si_inpaint \
       --csv figures/metrics_cross_re${re}_raw.csv
   done
   ```
   Compare each against the existing `figures/metrics_cross_re<N>.csv`. Geometry
   verified to match: both are `multiple 25`, so the comparison is like-for-like.

3. **Verify `corrlen = 14` px.** STILL UNVERIFIED and now load-bearing in three
   chapters plus the geometry table's `depth / corrlen` column. It predates
   `plot_cross_re_data.py`, which computes it two ways. Also check whether the
   code subtracts the mean before correlating — the thesis says "multiply their
   vorticity values", which is the raw correlation, not the covariance. For
   zero-mean vorticity they coincide, but confirm.

---

## 2. WHAT HAPPENED THIS SESSION

### Cluster
* **Ensemble analysis RAN** (job 5765518, 34 s on a MIG node) and the result is
  written up. See §3.
* **U-Net training confirmed complete** at epoch 2000 — the previous handoff had
  it as "still training".
* Job IDs decoded from runtimes: three `si_inf` at ~117 min are the ensemble
  runs (8 passes), four at ~15 min are the raw transfers. The 8:1 ratio confirms
  the mapping.
* **Every U-Net job is named `fno_inpaint` in Slurm** — both backbones share
  `run_train_fno_inpaint.sh`. Five jobs share that name; only the 26 h one is
  the U-Net training.

### Chapter 3 (methods) — now 2,011 words, **0 todos**
Cut from ~2,700 and restructured throughout:
* Two algorithms added, both transcribed from the code, not the equations:
  `alg_si_training.tex` and `alg_si_sampling.tex`. **Training has no integration
  loop** — one random tau per sample; the Heun loop is inference only.
* Architecture figure `fig_si_architecture.tex`, built after verifying every
  channel count by instantiating the model (7-agent workflow, all claims
  CONFIRMED). Plain-language labels; module names live in the appendix.
* Adversarial-objective figure `fig_adversarial.tex` — the patch critic's
  31x31 verdict grid, receptive field 46 px computed not guessed.
* Data-split figure `fig_data_split.tex`.
* **Wasserstein-1 DROPPED in favour of KL.** W1 moves 17.8% between replicates
  (78.5% worst case), KL moves 10.4%. The noise-floor sentence was updated from
  17.8% to 10.4% to match — it had been quoting the metric that was removed.
* `johnson2016perceptual` added to the bibliography (verified via Crossref) for
  the perceptual loss. Bibliography now 27 entries, all cited, none missing.
* Sections shortened hard: baselines 324 -> 182, evaluation protocol 411 -> ~330
  with the metrics as a list, implementation reduced to a Compute section.

### Chapter 2 (background) — 1,099 words, 0 todos
* **Score-based diffusion section DELETED** along with three orphaned citations.
* Four image-inpainting canon citations removed once their paragraph was cut.
* Terminology unified: `coverage` (defined at first use as the fraction
  REMOVED), `void` (not hole — 16 substitutions), `mask` kept distinct as the
  binary array.

### Chapter 4 (results) — 2,494 words, 4 todos, retitled to **Results**
Structure is now seven sections; the four baseline comparisons became run-in
`\paragraph`s inside "What governs the error" so their labels survive:
```
The crossed design / Qualitative results / What governs the error
  (depth law, classical floor, FNO baseline, objective control)
Sample diversity / External benchmark / Objective ablation
```
Written this session: crossed design, qualitative results, what governs the
error, sample diversity, depth law, classical floor, FNO baseline.

---

## 3. THE ENSEMBLE RESULT — READ THIS BEFORE WRITING ANYTHING ELSE

**The ratio FALLS with depth in all three runs.** The pre-registered expectation
in the section's own todo was that it would RISE.

| | edge | centre |
|---|---|---|
| single 25% (64 px) | 0.347 | 0.081 |
| single 6.25% (32 px) | 0.428 | 0.131 |
| multiple 25% (32 px) | 0.395 | 0.162 |

Mechanism: **spread saturates at roughly one correlation length** (peaks at 13,
13 and 19 px = 0.9–1.4 corrlen) while error keeps climbing to 4.5 corrlen. At
the centre of the largest void the eight samples sit **twelve times closer to
each other than to the truth**.

**Consequence, and it changes an interpretation elsewhere:** deep-void error
CANNOT be read as the model correctly identifying unrecoverable content. It is
confidently wrong. The "positive measurement of unrecoverability" framing the
previous handoff hoped for is unavailable. `sec:inp-ensemble` is written and
says so.

Note the ensemble runs use the **plain** recipe (`noema_noxshift`), so compare
against the plain rows, not `ema+xshift`.

---

## 4. NEW GENERATED ARTEFACTS (do not hand-edit)

All read the CSVs/npz directly, so numbers cannot drift from the data:

| script | output |
|---|---|
| `make_master_table.py` | `Thesis/figures/tab_master.tex` — every method x 9 configs, relL2 |
| `make_stats_table.py` | `Thesis/figures/tab_stats.tex` — physical metrics at single 25% |
| `make_spectra_figure.py` | `Thesis/figures/spectra_si_vs_fno.pdf` — in-void E(k), SI vs FNO |
| `make_depth_law_figure.py` | `Thesis/figures/depth_law.pdf` — profiles before/after dividing by edge |

`plot_si_fields.py` gained `--per-row N`. All four field figures regenerated:
truth+input on their own row, then N per row. **When the U-Net lands the panel
count goes 10 -> 12, so use `--per-row 5`** (2+5+5); at 4 it becomes eight rows
and overflows the page.

---

## 5. FINDINGS WORTH NOT LOSING

* **The FNO does not clear the classical floor.** Worse than the best classical
  rung in 3 of 9 cells; ties in most of the rest. A 9.5M-real-float trained
  operator performs like parameter-free interpolation. This is what makes the
  U-Net control necessary.
* **Biharmonic reverses the ranking.** Worst relL2 of any method at 64 px
  (1.139, worse than predicting nothing) and the BEST distributional scores of
  any method including ours, plus the brightest error-map peaks. The
  perception–distortion trade-off running in the direction that flatters the
  classical method. This is the empirical justification for the dual metric set.
* **The interpolant's advantage SHRINKS with depth** — 17.3x over the best
  classical rung at 8 px, 1.4x at 64 px. Opposite of what a reader expects; the
  chapter states it plainly.
* **In-void spectra:** at 64 px the FNO holds 0.06 of the true energy at its
  worst wavenumber, the SI 0.37. Both recover towards 1 at the highest k.
* **Depth law verified:** exponent 0.79–1.04, R^2 0.86–0.95, spread collapsing
  11–23x to 1.7–2.9x. Fractional depth is far WORSE (98–173x) — stated so it is
  not retried.
* **The todo's "4x coverage moves relL2 by ~10%" is only true at 32 px.** At
  8 px the same test gives 2.2x. The section reports both and draws the weaker,
  defensible claim. Do not quote the 10% alone.

---

## 6. STILL TO WRITE (4 todos in ch4, 22 in the thesis)

* `sec:inp-unet` — blocked on §1.1. The three outcomes are pre-committed in the
  todo; honour that framing.
* `sec:inp-shu` — data all local (centre-trained controls + their Table 1 in
  `eval_si_inpaint.SHU_TABLE1`). Nothing blocks this.
* `sec:inp-aux` — data local. GAN is the only resolvable win (variance +2.4%,
  enstrophy +4.9%, spectral -11%); p+g's -5.1% relL2 is INSIDE the 6.4% noise.
* Chapter-level todo at the top of 05_inpainting.tex.
* **Chapter 5 (generalisation), 6 (conclusions), both appendices.** Appendix A
  now owes a lot: architecture trace, naming/guard discipline, the incident
  catalogue, the coefficient-sum bug, the square-void refusal, the adaptive-
  weight verification. Appendix B assumes a CSV->LaTeX generator that does not
  exist — though `make_master_table.py` is now a working pattern for it.

---

## 7. WORKING AGREEMENTS (unchanged, plus one)

* Never commit or push on this project; end every file edit with the exact
  copy-pasteable command.
* `git pull` in `Thesis/` before editing — Overleaf is the other writer.
* **NEW, learned twice today: do not edit a passage in Overleaf while asking for
  it to be rewritten here.** Both edits are reasonable and git cannot tell which
  was meant; it produced two merge conflicts in one session.
* Wrap Overleaf git calls in a hard timeout (`perl -e 'alarm 90; exec @ARGV'` —
  macOS has no `timeout`).
* rsync to the cluster with an EXPLICIT file list, never a directory.
  Repo is at `/dss/dsshome1/0F/di24lir/Diffusion-based-Fluid-Super-resolution`;
  ckpts at `/dss/dssfs04/lwp-dss-0002/pn46yo/pn46yo-dss-0000/di24lir/ckpts`.
  Run transfers FROM THE MAC — running them in the cluster shell silently
  copies a file onto itself.
* Never poll `squeue`/`sacct` in a loop. ntfy topic `kaya-si-7h3k9x`.
* No local TeX toolchain — Overleaf is the only compile check.
* Verify every citation locator and every number from the todos before writing
  it. Two todo claims were wrong this session.
