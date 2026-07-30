# Results chapter — skeleton

One claim per section. Each section names the table or figure that carries it and the
runs behind it. If a section has no artifact, it is not yet a result.

Status legend: **[READY]** artifact exists · **[NEEDS FIG]** numbers exist, figure doesn't ·
**[BLOCKED]** waiting on data transfer.

---

## 5.1 Setup and what counts as success

**Claim:** the task is 1024 random sensors → 256² vorticity on 1272 held-out frames, and it
must be judged on four axes at once — field accuracy, physics consistency, spectral fidelity,
and usability downstream. No single scalar is sufficient, and the chapter is organised around
that fact.

- Table: benchmark configuration (Re=1000 Kolmogorov, 40 sims × 320 frames, train sims 0–35 /
  test 36–39, 1.6% sensor coverage).
- Point to make early: the 40 sensor layouts are all distinct, so unseen-position
  generalisation is baked into the dataset — **density**, not position, is the variable.
  (Pre-empts a reviewer question and stops a non-result being claimed as one.)
- Metrics defined here once: MSE / corr / std%, NS residual, e_spec / Δs / KE% / Z%.
  Flag that `l2_loss` in the logs is per-frame RMSE on unscaled vorticity, and that the
  chapter quotes `metrics.py`'s MSE, not the squared log value.

## 5.2 Headline comparison

**Claim:** SI wins decisively on every axis simultaneously — MSE 1.449 vs 3.431 for the best
diffusion variant, residual 8.28, e_spec 0.008.

- **[READY]** `metrics_table.txt` — MSE / RMSE / corr / std%
- **[READY]** `metrics_table_jcp.md` — JCP-style table for comparability with Shu et al.
- **[READY]** `stats_baseline_dps_si_vs_reference.png` — E(k) + p(ω)
- **[READY]** `spectrum_ratio_3runs.png` — the ratio plot; say explicitly that the log-log
  E(k) overlap is a *forgiving* metric and the ratio plot is what discriminates.
- **[READY]** `energy_map_kinetic_*.png`, `energy_map_enstrophy_*.png`

## 5.3 The framing: how much of the measurement survives into generation

**Claim:** one design decision — whether the measurement is destroyed before generation —
explains both the accuracy ranking and the noise ranking. Baseline/DPS noise at t=400 destroy
80% of the measurement: noise-immune (2.4% / 2.9% pass-through) but less accurate. SI starts
*at* the measurement: accurate, but passes 116% of sensor noise.

This is the section that turns the chapter from a leaderboard into an argument. Everything
after it is a consequence, not a new fact.

- **[NEEDS FIG]** the pass-through figure from `noise_robustness.py` — fraction of injected
  sensor noise surviving to output, three methods.
- Crossover at σ ≈ 0.31 (31% sensor error); real instruments are 1–5%. SI at 5% noise
  (MSE 1.517) still beats the baseline at *zero* noise (3.642).
- This closes the "SI never trained on noisy sensors" deployment objection without retraining.

## 5.4 Did we give diffusion a fair chance? (19 configurations)

**Claim:** no diffusion variant closes the gap. Every knob was tried; all land in MSE 3.4–4.2.
The SI result is earned, not an artifact of an under-tuned baseline.

Sub-claims, each a table row:

| sub-claim | evidence |
|---|---|
| Physics *conditioning* is inert | 0.03% effect, still inert at w=3.0 |
| `- dx` subtraction is the live pathway | costs ~3% MSE, buys 6.7× lower residual |
| JCP's iterative refinement hurts here | MSE 3.64 → 3.92 → 4.16, variance 84 → 77 → 75% |
| Lower t is artifacts surviving, not detail | t=200 variance 92.6% but residual triples to 173 |
| Sampler resolution was unfair, and fixing it doesn't help accuracy | r=20→100: residual 62.8 → 8.85, MSE 3.64 → 3.76 |
| DPS's residual problem is structural | λ sweep best 2536 vs target 12.5 — stop tuning |
| Input pre-smoothing is a trade, not a fix | see 5.5 |

**Honesty note to write explicitly:** quote baseline r=100 as the fair comparator. At matched
sampler resolution SI's *physics* edge nearly vanishes (8.28 vs 8.85) while its *accuracy*
edge (2.5×) survives intact. Claiming both would overstate the result.

## 5.5 Every low-residual route is over-smoothing in disguise

**Claim:** low NS residual is not a proxy for good physics — three independent routes to a low
residual all turn out to be over-smoothing, and only the spectral metrics reveal it.

- **[READY]** `_sm7` input smoothing: MSE 3.642 → 3.454 and residual 62.6 → 19.5 look like a
  free win, but e_spec 0.306 → 0.318, Δs 0.24 → 0.54, Z% 70.8 → 64.7. Bought by smoothing.
- **[READY]** baseline r=100: residual → 8.83 but e_spec *worsens* to 0.377 and KE falls to 62%.
- **[READY]** SI on `lowpass:4`: residual 1.41, far *below* ground truth's own 12.5.
- Ground truth's own residual is 12.5. Anything meaningfully below it is suspect by construction.
- **[NEEDS FIG]** worth one small figure: residual vs e_spec scatter across all runs, showing
  the two are not monotonically related. Cheap and it makes the point instantly.

## 5.6 Blind augmentation fails, and the mechanism is informational

**Claim:** training SI across a distribution of degradations does not buy sparse-sensor
robustness. It shifts the curve up without flattening it — degradation rate 1024→256 is
2.40× (specialist) vs 2.39× (blind), identical.

- **[READY]** `si_robustness_curve.png`
- **[READY]** `degradation_families.png`
- Killer detail: blind *saw* 256 and 512 in training; the specialist never did, and still wins
  there. Capacity dilution beats degradation coverage.
- Interpretation: the rate at which accuracy falls with sensor count is a property of the
  **information available**, not of training coverage. You cannot augment your way out of
  missing information.
- Correct the over-strong framing here: SI does **not** require paired data — blind training
  uses GT only. The real requirement for both SI and DPS is *can you simulate the measurement
  operator A?* The baseline needs no model of A at all. State the deployment hierarchy plainly.
- Blind's one consistent edge: lower residual at all five densities. Real, modest, not an
  accuracy win.

## 5.7 Does the advantage survive the dynamics?

**Claim:** yes. Used as initial conditions for a solver restart, SI's advantage persists —
and the NS residual *overstates* DPS's harm.

- **[READY]** `reanalysis_divergence.png`
- Solver validated first: advances truth frame j → j+1 at 0.66% rel-L2 (persistence 12.9%),
  independently confirming the residual's Re / drag / forcing match the data generator.
- At +8 frames: truth floor 0.010, SI 0.273, DPS 0.414, baseline 0.439.
- **The surprise, and worth stating as one:** DPS does not blow up. Viscosity dissipates its
  spurious high-k within about a frame, but nothing restores the baseline's missing 30% energy.
  Spurious fine scales are dynamically *transient*; an energy deficit is *permanent*.

## 5.8 Does it generalise off the training Reynolds number?

**Claim (provisional):** the Re sensitivity is a property of the shared diffusion prior, not
of the guidance — baseline and DPS degrade identically — and SI is lower in absolute error at
every Re tested.

- **[BLOCKED]** needs `cross_re1000/si_...` and `cross_re2000/dps_..._z3.0`; the first is SI's
  own denominator, so the relative-trend column can't be completed without it.
- Numbers so far (mean L2 / residual), read **relative to each method's own Re1000**:
  Re500 1.508/47.2 · 1.500/1919 · 1.103/2.88 — Re1000 1.686/59.8 · 1.678/2415 · — —
  Re2000 1.998/67.3 · — · 1.655/2.60 — Re10000 2.406/80.6 · 2.426/2928 · 2.179/3.32
- Baseline vs DPS relative to Re1000: 0.894/1.185/1.427 vs 0.894/–/1.445. Identical.
- **Caveat that must appear in the text:** the `kf_vort_Re*_N256` family is not the generator
  the models trained on (std 4.16 vs 4.85, different forcing amplitude). Absolute scores are
  not comparable to published numbers; only the trend is meaningful.

## 5.9 Controls and threats to validity

**Claim:** the result is not an artifact of information leakage, a favourable sampler, or a
flattered training setup.

- **[READY]** initialisation ablation (t=1000, r=50): MSE 31.33, corr 0.173, vs 3.64 at t=400.
  Give the exact decomposition — MSE = 15.02 + 22.69 − 6.38 = 31.32 — and note this is an
  *initialisation-only* ablation, not a true null: `- dx` still delivers the measurement every
  step, which is why corr is 0.17 rather than 0. Independently corroborates 5.4's finding that
  `- dx` is the live pathway.
- **Every methodological asymmetry runs against SI:** no EMA (verified — neither checkpoint has
  a `model_raw` key), stride-4 training (4× fewer unique samples), repo UNet not ConvNeXt.
  The diffusion baseline *does* use EMA. So the result is understated, not flattered. Cheaper
  and more honest to say this than to spend ~15 h retraining.
- Deviations from Schiødt et al., all deliberate: vorticity not velocity, no Helmholtz–Hodge
  projection (it is a velocity-field operation), full-field not patch-based, sensor input not
  lowpass. Cross-reference SI_README §10.1–10.3, §7.6–7.7.
- **[BLOCKED]** `_mine` / `_mine_xshift` reproduction check.
- Supplementary: paper reproduction chain vs their Table 1.

## 5.10 Synthesis

**Claim:** restate 5.3 as the conclusion it earned. Measurement-destroying methods buy noise
immunity and pay in accuracy; measurement-preserving transport makes the opposite trade and
wins at every realistic noise level. The open question is not which method is better but
whether the measurement operator can be faithfully simulated — and that limit binds SI and
DPS equally while leaving the baseline free.

---

## Writing order (dependency-ordered, not chapter-ordered)

1. **5.2** — the tables exist; writing them up fixes notation and metric definitions for
   everything else.
2. **5.3** — the framing. Do this second: every later section refers back to it, and drafting
   it early stops the chapter drifting into a leaderboard.
3. **5.4 + 5.5** — the fairness case. Largest section, all artifacts ready.
4. **5.6 + 5.7** — the two strongest independent probes; both have their headline figure.
5. **5.9** — controls. Write while the numbers are fresh.
6. **5.8** — last, so the blocked transfer isn't on the critical path. If the two folders never
   arrive, it degrades gracefully to a baseline-vs-DPS claim, which is still a real finding.
7. **5.1** and **5.10** — bookends, written once the middle is settled.

## Outstanding figures (all cheap, all laptop-only)

- pass-through bar chart for 5.3 (`noise_robustness.py` already computes it)
- residual vs e_spec scatter for 5.5
- cross-Re trend lines for 5.8 (once unblocked)
