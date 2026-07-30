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

- **[READY]** `noise_passthrough.png` (from `noise_passthrough_plot.py`) — share of injected
  sensor-noise variance reaching the output. baseline 2.2/2.4%, DPS 3.0/2.9%, SI 123/119%
  at σ = 0.02/0.05. The ratio is **stable across both noise levels**, so it is a property of
  the method, not an artifact of one σ — worth one sentence.
- Definition to state in the text: pass-through = [MSE(σ) − MSE(0)] / (σ·scale)², with
  scale = 4.7869, the training-split std the runtime scaler uses
  (`data_scale = np.std(ref_data[:-4])`). `--meas_noise` is in standardized units.
- NOTE the number changed: earlier drafts said 116%, computed with scale 4.85 (the raw-data
  std). 4.7869 is what the code actually uses, giving 119.3%. Quote **119%**, not 116%.
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

**STATE THE MECHANISM HERE, ONCE, AND REFER BACK TO IT** (measured in §5.9, not inferred):
the residual applies derivative operators, so it weights by ~k². The three baseline
checkpoints differ *only* above k≈80 — a band holding **0.0000% of the total energy**. At
k=127 their E(k)/E_ref ratios are 26.2 / 37.1 / 26.4, which **orders their residuals exactly**
(62.8 / 85.0 / 63.9). So an energetically negligible difference moves the residual 35% while
MSE, corr, std, e_spec, Δs, KE and Z all stay within 0.2%. Every energy-based metric is blind
to the band the residual is dominated by. That one fact explains all four residual anomalies:
  - uncorrelated with e_spec across methods (§5.5)
  - overstates DPS's harm under dynamics (§5.7)
  - unstable under training variance alone (§5.9)
  - drops *below* ground truth's own 12.5 whenever a method over-smooths (below)

- **[READY]** `_sm7` input smoothing: MSE 3.642 → 3.454 and residual 62.6 → 19.5 look like a
  free win, but e_spec 0.306 → 0.318, Δs 0.24 → 0.54, Z% 70.8 → 64.7. Bought by smoothing.
- **[READY]** baseline r=100: residual → 8.83 but e_spec *worsens* to 0.377 and KE falls to 62%.
- **[READY]** SI on `lowpass:4`: residual 1.41, far *below* ground truth's own 12.5.
- Ground truth's own residual is 12.5. Anything meaningfully below it is suspect by construction.
- **[READY]** `residual_vs_spectrum.png` (from `residual_vs_spectrum_plot.py`) — all 40 runs
  as (residual, e_spec), both log. The two axes are close to uncorrelated.
- **THE STRONGEST SINGLE POINT ON THIS FIGURE, found 2026-07-30:** DPS has the worst residual
  of anything here (3179, i.e. 50× the baseline's 62.8) and yet a **5× better spectrum**
  (e_spec 0.059 vs 0.306). Meanwhile the baseline family's residual ranges over 50× —
  8.85 (r=100) to 420 (no physics) — while its e_spec barely moves (0.18–0.48). So within
  the diffusion family the residual is almost uninformative about spectral fidelity, and
  across families it points the wrong way. This is the same conclusion §5.7 reaches from the
  dynamics (spurious high-k is transient, an energy deficit is permanent) arrived at by a
  completely independent route — say so explicitly; two independent routes to one conclusion
  is much stronger than either alone.

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

**Claim:** SI's *accuracy* advantage transfers across Re; its *spectral* advantage does not.
The two axes fail in different ways, and that asymmetry is the finding.

- **[READY]** `cross_re_trend.png` (from `cross_re_plot.py`) — two panels, 10 of 12 runs.
  The script skips missing runs and marks them ×, so re-running fills gaps in.
- **Field error, normalised by each Re's own reference std** (see correction below):
  baseline 0.373 / 0.385 / 0.443 / 0.505 · DPS 0.372 / 0.383 / — / 0.509 ·
  SI 0.273 / — / 0.367 / 0.458
- Baseline and DPS are **indistinguishable** at every Re (0.373 vs 0.372, 0.505 vs 0.509) →
  the Re sensitivity is the shared diffusion prior, NOT the guidance mechanism.
- SI is lowest at every Re, so the accuracy advantage survives distribution shift — **but its
  slope is steeper**: SI degrades 1.68× over 500→10000 against baseline's 1.35×, so the edge
  erodes from 27% better at Re500 to 9% better at Re10000. State this; it bounds the claim.
- **`e_spec` is FLAT in Re for all three methods** (baseline ~0.34–0.37, SI ~0.245–0.259,
  DPS ~0.075–0.109). Accuracy degrades, spectral fidelity doesn't — two different failure modes.
- **THE HONEST LIMITATION, and the most important thing in this section:** on this
  off-distribution data SI's e_spec is **0.245**, versus **0.008** on the main benchmark — a
  30× collapse — and **DPS beats it 2.7×** (0.089). SI's spectral dominance is entirely
  in-distribution. Mechanism: SI is a specialist that reproduces the spectrum it was trained
  on even when the true one differs; DPS is measurement-guided over a generic prior, so it
  tracks whatever the data says. This is the SAME specialist-vs-general mechanism as §5.6,
  now on DISTRIBUTION shift rather than degradation shift — a second independent instance.
- **NORMALISATION CORRECTION (changes the numbers):** the logs' `mean l2 loss` is per-frame
  RMSE in absolute vorticity units, and the reference std GROWS with Re — 4.0386 / 4.3797 /
  4.5100 / 4.7625 (measured over all 64 batches). About a third of the raw degradation was
  the reference's own variance. Quote the normalised trend, not raw L2.
- **Caveat that must appear in the text:** the `kf_vort_Re*_N256` family is not the generator
  the models trained on — its Re1000 reference std is 4.3797 vs the main experiment's 4.763,
  a 9.4% gap (earlier notes said ~14%; that was wrong). Absolute scores are not comparable to
  published numbers; only the within-method trend is meaningful.
- **[RESOLVED — all 12 runs now in]** The falsifiable test came out on the DISTRIBUTION-shift
  side. **SI's e_spec at cross_re1000 is 0.226**, against **0.008** on the main benchmark —
  a 28× collapse at the *same nominal Reynolds number*. So the loss of spectral fidelity is
  caused by the change of generator (different forcing amplitude), NOT by Reynolds number.
  That is the cleanest available statement of SI's specialism, and it was a prediction made
  before the data arrived.
- Full normalised field error, and the erosion of SI's advantage:

  | Re | baseline | DPS | SI | SI vs baseline | SI rel. own Re1000 |
  |---|---|---|---|---|---|
  | 500 | 0.3733 | 0.3715 | 0.2730 | **26.9% better** | 0.893 |
  | 1000 | 0.3850 | 0.3832 | 0.3056 | 20.6% better | 1.000 |
  | 2000 | 0.4430 | 0.4427 | 0.3670 | 17.2% better | 1.201 |
  | 10000 | 0.5053 | 0.5093 | 0.4575 | **9.5% better** | 1.497 |

  SI wins field accuracy at every Re across a 20× range — but the margin erodes monotonically
  26.9 → 20.6 → 17.2 → 9.5%, and SI's own degradation (1.497×) is steeper than the baseline's
  (1.312×) or DPS's (1.329×). It also *gains* more at Re500 (0.893 vs 0.970). Steeper in both
  directions is exactly what a specialist looks like.
- e_spec by method: baseline 0.336–0.370, SI 0.226–0.259, **DPS 0.075–0.114 — best at every
  Re**. DPS's measurement guidance tracks whatever spectrum the data has; SI reproduces the one
  it was trained on. Flat in Re for all three.

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
- **[READY]** reproduction check — retraining the baseline from scratch reproduces the provided
  checkpoint to **within 0.2% on every metric**:

  | | MSE | corr | std/ref% | e_spec | Δs | KE% | Z% |
  |---|---|---|---|---|---|---|---|
  | provided weights | 3.6419 | 0.9196 | 84.1 | 0.306 | 0.24 | 69.4 | 70.8 |
  | retrained here | 3.6485 | 0.9194 | 84.1 | 0.307 | 0.23 | 69.3 | 70.7 |
  | retrained + x-shift | 3.6341 | 0.9198 | 84.1 | 0.304 | 0.27 | 69.6 | 70.7 |

  So every baseline number in the thesis rests on a reproducible artifact, not on a checkpoint
  that happened to be handed over. This is the row that makes the comparison trustworthy.
- **Clean negative result:** x-translation augmentation buys nothing — 3.6341 vs 3.6485 vs
  3.6419 are all inside run-to-run noise. The symmetry is free to exploit and it does not help.
  Report it; a negative result on a cheap idea is worth a sentence.
- **NEW, and it belongs in §5.5 as well:** the NS residual swings **62.56 → 84.59 (+35%)**
  between two training runs that agree to 0.2% on MSE, corr, std, e_spec, Δs, KE and Z.
  (x-shift lands at 63.60.) So the residual has poor run-to-run reproducibility while every
  other metric is stable — a third independent reason not to treat it as *the* physics metric.
  Cross-reference this from §5.5; it arrives from training variance rather than from sampler
  or method choice, so it is genuinely independent evidence.
- **[READY] Supplementary: the paper reproduction succeeded** (their code, their DNS, their
  ConvNeXt, their metrics, 100-step sampler — job 5834217 + a re-run with the analysis flags on).
  Artifacts in `paper_repro_plots/`:
  - **dissipation**: low-res input 0.759 of truth, SI reconstruction **0.996** — recovered to
    within 0.4%. (Their metric is the mean of per-sample ratios, not the ratio of means —
    state that, since 10.165/9.735 = 1.044 and someone will check.)
  - **`spectrum_ensemble_avg.png`**: the input departs from truth at k≈6 and falls orders of
    magnitude; SI tracks truth across the resolved range with a small excess past k≈40.
  - **`vorticity_field.png`**: input smooth and blobby, SI recovers the filamentary structure.
  - **max |∇·u| = 0.0296** — the quantity the Helmholtz–Hodge projection exists to control.
- **Why this row matters:** it separates two things an examiner would otherwise conflate. The
  port to the sparse-sensor benchmark deviates in four documented ways (vorticity not velocity,
  no Helmholtz projection, full-field not patch, sensor input not lowpass). This shows the
  *method* reproduces on its home ground, so any difference in our results is attributable to
  the transplant, not to a broken implementation. It also puts a number on the projection we
  dropped, which was previously the least-quantified of the four deviations.

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
