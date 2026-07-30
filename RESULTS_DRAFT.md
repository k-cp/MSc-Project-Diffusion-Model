# Chapter 5 — Results

> **Draft, 2026-07-30.** Every number here is measured, not estimated, and each section names
> the artifact that carries it. Placeholders that still need your input are marked `[TODO]`.
> Citation keys are written as `[Shu2023]`, `[Chung2023]`, `[Schiodt2026]`, `[Ren2026]` —
> replace with your bibliography's keys.

---

## 5.1 Setup, and what counts as success

All three methods are evaluated on the sparse-sensor benchmark of Shu et al. `[Shu2023]`:
two-dimensional Kolmogorov flow at Re = 1000, forced by −4 cos(4y) with linear drag, on a
256 × 256 grid. The dataset comprises 40 independent simulations of 320 frames each.
Simulations 0–35 form the training split and simulations 36–39 are held out, giving 1272
evaluation samples once each frame is taken as the centre of a three-frame sliding window.
The three-frame window is required by the vorticity-transport residual, which needs a temporal
derivative; only the middle frame of each window is scored.

The measurement is 1024 randomly positioned point sensors — **1.6% of the grid** — filled to
full resolution by nearest-neighbour interpolation. One property of the dataset deserves
stating early, because it forecloses a claim this work does *not* make: each of the 40
simulations carries its own independent sensor layout. Position-invariance is therefore a
property of the dataset construction, not an achievement of any model, and the held-out
simulations already test unseen sensor positions by default. The variable this chapter
explores is sensor **density**, not sensor placement.

### Metrics

No single scalar is adequate here, and the organisation of this chapter follows from that.
Four families are reported throughout:

| family | metric | reads |
|---|---|---|
| field accuracy | MSE, RMSE, correlation | pointwise closeness to truth |
| variance | std / std<sub>ref</sub> | over-smoothing (< 100%) or spurious content (> 100%) |
| physics | Navier–Stokes residual | consistency with the governing equation |
| spectrum | e<sub>spec</sub>, slope deviation Δs, KE%, Z% | *what kind* of turbulence was produced `[Ren2026]` |

Two conventions matter for reproducing these numbers. First, the per-batch quantity written to
the run logs as `mean l2 loss` is a per-frame **RMSE** on unscaled vorticity, averaged over
frames — not an MSE. All MSE figures in this chapter are computed from the saved arrays by
`metrics.py`, and the two differ slightly by Jensen's inequality (1.874² = 3.51 against a
measured 3.642). Second, the ground truth's own discretised residual is **12.5**. That number
is the reference against which every "low residual" claim in this chapter must be read, and
Section 5.5 shows that values far *below* it are a symptom rather than an achievement.

---

## 5.2 Headline comparison

Table 5.1 gives the complete comparison on the full test set.

**Table 5.1** — All methods, 1272 held-out frames. Best value in each column in bold.

| method | MSE | RMSE | corr | std/ref % | residual | e<sub>spec</sub> | slope | Δs | KE % | Z % |
|---|---|---|---|---|---|---|---|---|---|---|
| ground truth | — | — | — | 100.0 | 12.5 | 0.000 | −5.13 | 0.00 | 100.0 | 100.0 |
| Diffusion baseline `[Shu2023]` | 3.642 | 1.908 | 0.9196 | 84.1 | 62.8 | 0.306 | −5.37 | **0.24** | 69.4 | 70.8 |
| DPS, ζ = 3.0 `[Chung2023]` | 3.589 | 1.894 | 0.9177 | 93.7 | 3179 | 0.059 | −4.84 | 0.29 | 94.1 | 87.8 |
| Stochastic interpolants `[Schiodt2026]` | **1.449** | **1.204** | **0.9678** | **99.2** | **8.28** | **0.008** | −5.52 | 0.39 | **100.3** | **98.4** |

Stochastic interpolants win seven of the nine scored columns. The margin on field accuracy is
large: MSE 1.449 against 3.589 for the best single diffusion configuration in this table, and
against **3.431** for the best of the nineteen diffusion variants examined in Section 5.4. On
variance the gap is starker still — SI reproduces 99.2% of the reference standard deviation
while the baseline reproduces 84.1%, and the integrated spectral error differs by a factor of
38 (0.008 against 0.306).

Two features of Table 5.1 complicate any simple ranking, and both are worth stating here
rather than leaving for a reader to find.

**SI loses on the cascade slope.** Δs is the deviation of the fitted log-log spectral slope
from the reference's −5.13 over k ∈ [10, 60]. SI's tail is too steep at −5.52, the baseline's
is closest at −5.37, and DPS's is too shallow at −4.84. The signs are informative: SI
over-smooths its tail, DPS injects noise into it, and the baseline — worst on energy content by
a wide margin at 69.4% KE — happens to preserve the *shape* of what little it retains. A method
that led every column would invite suspicion; the one column SI loses is the one that describes
the character of the turbulence rather than its magnitude.

**DPS beats the baseline on spectrum while losing catastrophically on residual.** DPS's
residual is 3179 against the baseline's 62.8, a factor of 50, yet its integrated spectral error
is five times *better* (0.059 against 0.306), it retains 94.1% of the kinetic energy against
69.4%, and it wins MSE. Any ranking constructed from the residual alone inverts this pair.
Section 5.5 takes up that observation and Section 5.9 explains its mechanism.

Figures `[TODO: insert]` `stats_baseline_dps_si_vs_reference.png` and `spectrum_ratio_3runs.png`
give the energy spectra and vorticity distributions. Note that the E(k) and p(ω) curves overlap
substantially for all three methods; these are forgiving presentations, and the discriminating
views are the ratio plot and the scalars in Table 5.1.

---

## 5.3 What survives into generation

The results in Table 5.1 follow from a single design decision, and reading them through it
turns a leaderboard into an argument.

The diffusion baseline and DPS both begin by noising the measurement to t = 400 of a
1000-step schedule, which destroys roughly 80% of its variance before generation begins.
Stochastic interpolants instead define a transport that begins *at* the measurement and ends at
the reconstruction: the measurement is retained in full. This is not a difference of degree.

The consequence is visible in an experiment designed to expose it. Gaussian sensor noise of
standard deviation σ (in standardised units) was injected into the measurement at inference
time, and the share reaching the output was computed as

> pass-through = [ MSE(σ) − MSE(0) ] / (σ · s)²,  s = 4.7869

where s is the training-split standard deviation used by the runtime scaler. **Figure 5.x**
(`noise_passthrough.png`):

| method | σ = 0.02 | σ = 0.05 |
|---|---|---|
| Diffusion baseline | 2.2% | 2.4% |
| DPS | 3.0% | 2.9% |
| Stochastic interpolants | 123.0% | 119.3% |

The pass-through is stable across both noise levels, so it is a property of the method rather
than an artifact of one perturbation size. The methods that destroy the measurement destroy the
sensor error with it and are almost perfectly noise-immune; the method that preserves the
measurement passes the error through essentially one-to-one.

**One decision, two consequences.** Destroying the measurement buys noise immunity *and* costs
accuracy — these are the same mechanism observed twice, not two independent findings. Every
result in the remainder of this chapter is a consequence of this trade.

The practical question is which side of it is preferable, and the answer is not close. At 5%
sensor noise SI achieves MSE 1.517, still 2.4 times better than the baseline achieves at **zero**
noise (3.642). The crossover at which SI's advantage would be erased is σ ≈ 0.31, i.e. a 31%
sensor error; real instrumentation operates at 1–5%. The noise sensitivity is real, and at any
realistic noise level it is irrelevant.

---

## 5.4 Was diffusion given a fair chance?

Because the headline claim is comparative, the baseline must be shown to have been tuned rather
than merely run. Nineteen diffusion configurations were evaluated across five axes. **All of
them land between MSE 3.4 and 4.2.** No configuration closes the gap to 1.449.

**Table 5.2** — Diffusion configurations. `[TODO: trim to the rows you want to show]`

| axis | configuration | MSE | residual | note |
|---|---|---|---|---|
| reference | t=400, r=20, ss=1, w=0 | 3.642 | 62.8 | as shipped |
| physics | conditioning only | 3.539 | 414 | |
| physics | linear (`− dx`) only | 3.643 | 63.3 | |
| physics | none | 3.540 | 420 | |
| sampler | r = 50 | — | 10.7 | |
| sampler | r = 100 | 3.758 | 8.85 | matched to SI's budget |
| refinement | sample_step = 3 | 3.92 | 40.2 | |
| refinement | sample_step = 5 | 4.16 | 99.1 | |
| noise level | t = 300 | — | 93.8 | |
| noise level | t = 200 | — | 172.8 | variance 92.6% |
| DPS + physics | λ = 0.05 | 3.563 | 2536 | best DPS residual |
| DPS + physics | λ = 1.0 | **3.431** | 4252 | best diffusion MSE |

Four results are worth drawing out.

**The physics conditioning is inert; the subtraction does the work.** The rows above pair
exactly by the presence of the `− dx` term. Conditioning the network on the physics gradient
changes MSE by 0.03% and remains inert even at guidance weight w = 3.0. The direct subtraction
costs about 3% of MSE and three percentage points of variance, but buys a 6.7-fold reduction in
residual. It is a genuine accuracy-physics trade rather than a redundant term — and MSE alone
would have mislabelled it as useless.

**The original comparison was unfair to the baseline, and correcting it matters for one claim
only.** The baseline as shipped takes 20 reverse steps against SI's 100, an integration
resolution difference of five-fold. Raising it to r = 100 reduces the residual from 62.8 to
**8.85**, essentially eliminating SI's physics advantage (8.28). But MSE does not improve — it
degrades slightly to 3.758. **SI's field-accuracy advantage survives the fairness correction
intact; its physics advantage does not.** The r = 100 figure is the honest comparator and is
used as such throughout.

**The mechanisms the original paper introduced hurt on this task.** Iterative refinement was
introduced by `[Shu2023]` specifically for sparse random sensors, with the stated expectation
that more iterations help as the input-target gap widens. At 1.6% coverage the gap is extreme,
yet MSE degrades monotonically (3.64 → 3.92 → 4.16) and variance falls (84.1% → 77.2% → 75.1%),
i.e. the output becomes *more* over-smoothed. Similarly, lowering the noise level appears to
recover variance (92.6% at t = 200) but this is measurement artifact surviving the reverse
process rather than detail being recovered: the residual triples to 173.

**DPS's residual problem is structural, not a tuning failure.** A physics gradient was added to
the DPS update and its weight swept. The best achievable residual is 2536 at λ = 0.05 — a 20%
improvement on a quantity that needs to fall by a factor of 200. Larger weights overshoot: the
physics gradient has magnitude ~15–25 against the measurement gradient's ~1.2. Measurement
guidance backpropagates through the network and injects high-wavenumber content that a physics
nudge cannot remove. This line of tuning was abandoned on that basis rather than exhausted.

---

## 5.5 A low residual is not good physics

The Navier–Stokes residual is the physics metric this literature quotes most often. Used alone
it is misleading, and this section establishes why — with a mechanism measured in Section 5.9
rather than inferred.

**Figure 5.x** (`residual_vs_spectrum.png`) plots all 40 evaluated runs as (residual,
e<sub>spec</sub>). The two axes are close to uncorrelated. Within the diffusion baseline family
the residual spans a factor of 50 — from 8.85 at r = 100 to 420 with physics disabled — while
e<sub>spec</sub> barely moves across the range 0.18–0.48. Across method families it points the
wrong way entirely: DPS has the worst residual on the chart and a five-fold better spectrum
than any baseline variant.

Three independent routes to a low residual all turn out to be over-smoothing:

- **Input pre-smoothing** (`_sm7`) improves MSE to 3.454 and cuts the residual to 19.5 — but
  e<sub>spec</sub> worsens (0.306 → 0.318), Δs more than doubles (0.24 → 0.54) and enstrophy
  falls from 70.8% to 64.7%. The gain is bought by smoothing harder.
- **Sampler resolution** (r = 100) reaches residual 8.83 while e<sub>spec</sub> *worsens* to
  0.377 and kinetic energy falls to 62.3%.
- **SI evaluated on a low-pass degradation** reaches residual 1.41 — nine times *below* the
  ground truth's own 12.5.

That last figure makes the point sharpest. The residual of the true field is 12.5; any method
reporting substantially less than that is not more physical than reality, it has removed the
small scales that the residual's derivative operators would have penalised.

### The mechanism

Section 5.9 measures the cause directly. The residual applies derivative operators and
therefore weights the spectrum by approximately k². Three independently trained baseline
networks agree to within 0.2% on MSE, correlation, variance, e<sub>spec</sub>, Δs, kinetic
energy and enstrophy, and differ **only above k ≈ 80** — a band containing **0.0000% of the
total energy**. Their residuals nevertheless differ by 35%. An energetically negligible
difference dominates the residual while remaining invisible to every energy-based metric.

That single fact accounts for all four anomalies observed in this chapter: the residual's
lack of correlation with spectral error (this section), its overstatement of DPS's harm under
dynamics (Section 5.7), its instability under training variance alone (Section 5.9), and its
tendency to fall below the ground truth's own value whenever a method over-smooths.

**Recommendation for practice.** The residual should be reported alongside a spectral measure,
never in place of one, and always against the ground truth's own value rather than against
zero.

---

## 5.6 Blind augmentation does not buy robustness

If SI's weakness is that it is a specialist, the obvious remedy is to train it across a
distribution of degradations. This was implemented — sensor sub-sampling with n ~ U[256, 4000],
plus down-sampling and low-pass families, generated on the fly from ground truth alone — and
evaluated against the specialist across sensor densities.

**Table 5.3** — Field error (L2) and residual, specialist against blind.

| density | specialist | blind | L2 winner |
|---|---|---|---|
| 256 | 2.889 / 8.19 | 3.112 / 7.33 | specialist |
| 512 | 1.884 / 7.36 | 2.119 / 7.30 | specialist |
| 1024 (fixed) | 1.170 / 8.28 | 1.268 / 7.79 | specialist |
| 1024 (random) | 1.202 / 8.35 | 1.303 / 7.81 | specialist |
| 2048 | 0.882 / 10.64 | 0.761 / 9.09 | blind |

The hypothesis is refuted, and the manner of its refutation is the finding.

**The blind model saw 256 and 512 in training; the specialist never did — and the specialist
still wins there.** Coverage of a degradation is not the same as competence at it.

**The degradation rate is identical.** Going from 1024 to 256 sensors costs the specialist a
factor of 2.40 and the blind model a factor of 2.39. Augmentation did not flatten the curve at
all; it shifted it upward by 0.10–0.24 in L2. The rate at which accuracy falls with sensor
count is a property of the **information available in the measurement**, not of training
coverage. One cannot augment one's way out of missing information.

Blind training's one consistent advantage is a lower residual at every density — small,
perfectly consistent, and (per Section 5.5) not an accuracy result.

**A correction to a claim made earlier in this work.** It is too strong to say that SI requires
paired low- and high-resolution data. Blind training uses ground truth only and manufactures
the measurement on the fly. The real requirement, shared with DPS, is that the measurement
operator *A* can be simulated. The deployment hierarchy is: the diffusion baseline needs no
model of *A* at all; DPS and both SI variants need one. Where *A* cannot be faithfully
simulated — real PIV optics, particle dynamics, correlated noise — SI and DPS are equally
exposed. The practical implication is that if one can simulate *A* at all, one can simulate the
specific *A* of interest, and the specialist should be preferred; blind training's generality
buys nothing here, since low-density loss is not a training-coverage problem.

Figures: `si_robustness_curve.png`, `degradation_families.png`.

---

## 5.7 Does the advantage survive the dynamics?

Field-error metrics score a static snapshot. A reconstruction intended for use — as an initial
condition for a solver, or as a reanalysis product — must also behave under the governing
equations.

A pseudo-spectral RK4 solver was implemented from the same formulation as the residual (same
wavenumber layout, 2/3 dealiasing, eight substeps per frame) and validated against the data: it
advances a true frame to the next with 0.66% relative L2 error, against 12.9% for persistence.
That agreement independently confirms that the residual's parameters (Re = 1000, drag 0.1,
forcing −4 cos 4y) match the data generator.

Reconstructions were then used as initial conditions across 12 restarts. Relative L2 error at
+8 frames:

| | truth (floor) | SI | DPS | baseline |
|---|---|---|---|---|
| rel. L2 at +8 frames | 0.010 | **0.273** | 0.414 | 0.439 |

SI's advantage survives the dynamics. But the ordering of the other two is the surprise, and it
contradicted the prediction made before the experiment: **DPS does not blow up.** Despite a
residual of 3179 it beats the baseline at every horizon, and its error even dips between +1 and
+4 frames.

The mechanism is physical. Viscosity dissipates DPS's spurious high-wavenumber content within
roughly one frame, whereas nothing restores the baseline's missing 30% of kinetic energy.
**Spurious fine-scale content is dynamically transient; an energy deficit is permanent.** For
restart applications the NS residual therefore substantially overstates DPS's harm — the third
independent demonstration in this chapter that the residual is a poor summary statistic.

Figure: `reanalysis_divergence.png`.

---

## 5.8 Generalisation across Reynolds number

All models were trained at Re = 1000 and evaluated zero-shot at Re = 500, 1000, 2000 and 10000
on an independently generated dataset.

**A caveat that governs the reading of this section, quantified.** The `kf_vort_Re*_N256` family
is not the generator the models were trained on, and the mismatch is not small. Characterising
each evaluation set against the training distribution (`inspect_cross_re_data.py`,
**Figure 5.x**, `cross_re_data.png`):

| dataset | std | e<sub>spec</sub> vs training data | energy in k ≤ 4 | fraction \|ω\| > 15 |
|---|---|---|---|---|
| training data | 4.7632 | — | 90.9% | 0.317% |
| Re = 500 | 4.0386 | 0.456 | 92.5% | 0.323% |
| Re = 1000 | 4.3798 | 0.426 | 90.8% | 0.427% |
| Re = 2000 | 4.5100 | 0.391 | 91.0% | 0.567% |
| Re = 10000 | 4.7625 | **0.343** | 91.1% | 0.820% |

**None of the four datasets is in-distribution**, and Re = 1000 is not even the closest — that is
Re = 10000, while Re = 500 is the most distant. The nominal Reynolds number therefore does not
identify the training condition in this family, which is why the spectral result below should not
be read against the benchmark value of Table 5.1. Note also that matching variance is not
matching distribution: the Re = 10000 set reproduces the training standard deviation to four
significant figures (4.7625 against 4.7632) while containing two and a half times as many
extreme-vorticity points. Absolute scores here are not comparable to Table 5.1; only the trend
within a method is meaningful.

**A normalisation that matters.** The reference standard deviation grows with Reynolds number
(4.0386, 4.3797, 4.5100, 4.7625), so raw error grows partly because the field has more variance
to get wrong. Roughly a third of the apparent degradation is this effect. All figures below are
normalised by each Reynolds number's own reference standard deviation.

**Table 5.4** — Normalised field error. **Figure 5.x** (`cross_re_trend.png`).

| Re | baseline | DPS | SI | SI's margin | SI rel. own Re1000 |
|---|---|---|---|---|---|
| 500 | 0.3733 | 0.3715 | **0.2730** | 26.9% | 0.893 |
| 1000 | 0.3850 | 0.3832 | **0.3056** | 20.6% | 1.000 |
| 2000 | 0.4430 | 0.4427 | **0.3670** | 17.2% | 1.201 |
| 10000 | 0.5053 | 0.5093 | **0.4575** | 9.5% | 1.497 |

**Is zero-shot transfer useful at all?** Both models were trained at a single Reynolds number
(Section `[TODO: xref training setup]`), so the relevant bar is the error of the raw measurement
itself. Table 5.4b adds it.

**Table 5.4b** — Each method against the measurement it was given. Same normalisation.

| Re | measurement | baseline | DPS | SI | SI's improvement on the measurement |
|---|---|---|---|---|---|
| in-distribution benchmark | 0.500 | — | — | **0.252** | **49.6%** |
| 500 | 0.464 | 0.373 | 0.371 | **0.273** | 41.2% |
| 1000 | 0.483 | 0.385 | 0.383 | **0.306** | 36.7% |
| 2000 | 0.537 | 0.443 | 0.443 | **0.367** | 31.6% |
| 10000 | 0.599 | 0.505 | 0.509 | **0.457** | 23.7% |

Every method improves on its measurement at every Reynolds number: nothing collapses, and no
configuration is worse than doing nothing. Transfer from single-Reynolds training across a
twenty-fold range is therefore genuinely useful, and SI is the best of the three at every point.

The erosion, however, is substantial, and it decomposes into two roughly equal contributions.
Changing generator at fixed Reynolds number costs about thirteen percentage points of recovered
error (49.6% → 36.7%), and the subsequent extrapolation from Re = 1000 to Re = 10000 costs
another thirteen (36.7% → 23.7%). The unintended distribution shift is as expensive as the
intended Reynolds extrapolation.

Note also that the task itself becomes harder with Reynolds number — the measurement's own error
rises from 0.464 to 0.599 as more of the flow's structure falls below the sensor spacing. But
SI's error rises faster than that (67% against the task's 29%), so the degradation is not merely
the problem getting harder; there is genuine extrapolation loss on top. The fraction of the
measurement's deficit that SI closes, falling from 41% to 24%, is the cleanest single statement
of it.

**The baseline and DPS are indistinguishable at every Reynolds number** — 0.3733 against 0.3715
at Re = 500, 0.5053 against 0.5093 at Re = 10000. Whatever sensitivity to Reynolds number these
methods have belongs to the shared diffusion prior, not to the guidance mechanism that
distinguishes them.

**SI's accuracy advantage transfers across a twenty-fold range of Reynolds number** — it is
lowest at every point tested. But the margin erodes monotonically, from 26.9% at Re = 500 to
9.5% at Re = 10000, and SI's own degradation (a factor of 1.497 across the range) is steeper
than the baseline's (1.312) or DPS's (1.329). SI also *gains* more than the others at Re = 500
(0.893 against 0.970). Steeper in both directions is the signature of a specialist: more upside
on easier data, more downside on harder.

### The spectral result, and a limitation

Integrated spectral error is **flat in Reynolds number** for all three methods — baseline
0.336–0.370, SI 0.226–0.259, DPS 0.075–0.114. Accuracy degrades with Re; spectral fidelity does
not. The two axes fail in different ways.

But the levels invert the headline. **SI's e<sub>spec</sub> here is 0.226–0.259, against 0.008
on the benchmark of Table 5.1 — and DPS beats it at every Reynolds number.** SI's spectral
dominance is entirely in-distribution.

The cause can be identified, and was predicted before the data were available. If the collapse
were a function of Reynolds number, SI's e<sub>spec</sub> at Re = 1000 — its training Reynolds
number — should have remained near 0.008. It is **0.226**, a 28-fold degradation at the *same
nominal Reynolds number*. The cause is therefore the change of generator and not Reynolds number
at all, which the dataset characterisation above corroborates: the Re = 1000 evaluation set sits
0.426 from the training distribution in the same spectral measure.

It should be said that the *ordering* across the four Reynolds numbers is not explained by
distance from the training distribution. SI is best at Re = 1000 (0.226) and worst at
Re = 10000 (0.259), while the datasets run in the opposite order (0.426 and 0.343
respectively). The defensible claim is the weaker one: all four evaluation sets are comparably
far from the training distribution, and SI's spectral error is correspondingly flat and
uniformly poor. No finer mechanism is claimed here.

The mechanism is the same specialism identified in Section 5.6, now observed under a different
kind of shift: SI reproduces the spectrum it was trained on, confidently, even where the true
spectrum differs. DPS, guided by the measurement over a generic prior, tracks whatever spectrum
the data actually has. This is the third appearance of one property — specialisation — after
the density sweep and the degradation-family results.

---

## 5.9 Controls, reproduction, and threats to validity

### The baseline is a fair opponent

The comparative claim depends on the baseline being competently trained. It was retrained from
scratch, with and without physics-valid x-translation augmentation.

**Table 5.5** — Three baseline checkpoints.

| checkpoint | MSE | corr | std/ref % | e<sub>spec</sub> | Δs | KE % | Z % | residual |
|---|---|---|---|---|---|---|---|---|
| provided weights | 3.6419 | 0.9196 | 84.1 | 0.306 | 0.24 | 69.4 | 70.8 | 62.8 |
| retrained | 3.6485 | 0.9194 | 84.1 | 0.307 | 0.23 | 69.3 | 70.7 | 85.0 |
| retrained + x-shift | 3.6341 | 0.9198 | 84.1 | 0.304 | 0.27 | 69.6 | 70.7 | 63.9 |

The reproduction agrees to within 0.2% on every energy-based metric. More strongly, the three
networks produce the *same field*, not merely the same score: pairwise MSE between
reconstructions is 0.044–0.046 with correlation 0.9986, against 3.63–3.65 against ground truth.
**They agree with each other roughly eighty times more closely than any agrees with the truth.**
Independent training runs converge to nearly the same function, so the baseline's error is a
property of the method rather than of a particular checkpoint.

**x-translation augmentation buys nothing.** The three MSE values lie within run-to-run noise.
The forcing −4 cos(4y) varies along one axis only, so translation along the other is exactly
symmetry-preserving and the augmentation is free — and it makes no difference.

The one metric that fails to reproduce is the residual, at 62.8 against 85.0, a uniform shift
rather than an outlier effect (median 61.8 → 81.1). Section 5.5 uses this: the three
checkpoints differ only above k ≈ 80, in a band holding 0.0000% of the energy, and their
E(k)/E<sub>ref</sub> ratios at k = 127 — 26.2, 37.1, 26.4 — order their residuals exactly.
Figure: `baseline_reproduction.png`.

### Information control

To confirm that the measurement carries genuine information at t = 400, the initialisation was
destroyed entirely (t = 1000, r = 50). MSE rises from 3.642 to **31.33** and correlation falls
to 0.173.

The decomposition is exact:

> MSE = σ²<sub>x</sub> + σ²<sub>y</sub> − 2ρσ<sub>x</sub>σ<sub>y</sub> = 15.02 + 22.69 − 6.38 = 31.32

against a measured 31.3256. Note that this is an **initialisation-only** ablation rather than a
true null: the `− dx` term continues to deliver the measurement at every reverse step, which is
why the correlation is 0.173 rather than 0. That residual correlation independently corroborates
Section 5.4's finding that the subtraction, not the conditioning, is the live pathway.

### Every methodological asymmetry runs against SI

- SI was trained **without EMA** (verified by inspecting both checkpoints), while the diffusion
  baseline uses it and loads the EMA state at inference.
- SI trained on a frame stride of 4, i.e. **four times fewer unique samples**.
- SI uses the repository's UNet rather than the ConvNeXt architecture of `[Schiodt2026]`.

Each of these disadvantages SI, and SI wins by a factor of 2.5 regardless. The result is
therefore understated rather than flattered.

### Independent reproduction of the source method

To separate "the transplant is faithful" from "the method works", the source method was
reproduced end to end using the authors' own code, solver, architecture and metrics
`[Schiodt2026]`, on their own DNS at 128², with their 100-step sampler.

| | mean dissipation | ratio to truth |
|---|---|---|
| low-resolution input | 7.235 | 0.759 |
| ground truth | 9.735 | 1.000 |
| SI reconstruction | 10.165 | **0.996** |

The input is missing a quarter of the dissipation and the method recovers it to within 0.4%.
(Their metric is the mean of per-sample ratios, not the ratio of means; the latter is 1.044.)
The ensemble-averaged spectrum shows the input departing from truth at k ≈ 6 and falling orders
of magnitude, with the reconstruction tracking truth across the resolved range. The maximum
absolute divergence of the reconstruction is 0.0296 — the quantity that the Helmholtz–Hodge
projection exists to control.

This matters because the port to the sparse-sensor benchmark deviates from `[Schiodt2026]` in
four documented respects: vorticity rather than velocity, no Helmholtz–Hodge projection (a
velocity-field operation, inapplicable to a vorticity formulation), full-field rather than
patch-based inference, and sensor measurements rather than low-pass filtering. The reproduction
shows the method performing as published on its home ground, so any divergence in the results
above is attributable to the transplant rather than to a defective implementation. It also puts
a scale on the projection that was dropped.

### Port against source, under one set of definitions

The two implementations can be compared once their outputs are reduced to dimensionless ratios
against their own ground truth. The source's velocity fields were converted to vorticity by a
spectral curl so that both pass through the same `compute_ke_spectrum` used throughout this
chapter (`compare_si_implementations.py`).

**Table 5.6** — Source implementation against this port. Raw errors are not comparable — the
tasks differ in field, resolution and degradation — so every entry is relative to its own truth.

| | source `[Schiodt2026]` | this port |
|---|---|---|
| task | 128², velocity, low-pass | 256², vorticity, 1024 sensors (1.6%) |
| input e<sub>spec</sub> | 0.039 | 0.053 |
| **output e<sub>spec</sub>** | **0.0024** | 0.0084 |
| spectral gap closed | **94.0%** | 84.2% |
| input norm. field error | 0.464 | 0.500 |
| output norm. field error | 0.591 | **0.252** |
| input correlation | 0.8864 | 0.8751 |
| output correlation | 0.8262 | **0.9680** |
| output enstrophy ratio | 1.012 | 0.984 |

**On integrated spectral error the source scores better** — 0.0024 against 0.0084, closing 94.0%
of the spectral gap against this port's 84.2%. The comparison is not confounded by resolution, as
might be feared: only 0.01% of this port's spectral error lies above k = 63, the source's highest
resolved wavenumber, so restricting both to the matched band k ≤ 63 leaves the figures unchanged.

**The scale-resolved picture is more informative than the scalar.** Energy reconstructed
relative to truth, per wavenumber (no binning, to avoid the averaging artifact noted below):

| k | 1 | 4 | 8 | 12 | 16 | 24 | 32 | 40 | 48 | 56 |
|---|---|---|---|---|---|---|---|---|---|---|
| source | 1.000 | 1.001 | 1.021 | 0.879 | 1.004 | 1.065 | 1.180 | 1.282 | 1.563 | 4.325 |
| this port | 1.003 | 0.994 | 0.997 | 0.972 | 0.895 | 0.811 | 0.723 | 0.636 | 0.597 | 0.524 |

Both reproduce the energy-containing scales (k ≤ 4) essentially exactly. They then diverge **in
opposite directions**: the source runs progressively over-energetic, this port progressively
over-smoothed. Across the physically meaningful small-scale range, roughly k = 16–45, **the
source is the closer of the two** — it stays within about 30% of truth where this port falls to
0.64. The port's monotone decline is the same over-smoothing that Table 5.1 records as Δs = 0.39,
and it is what an interpolant pinned by 1024 exact point values would be expected to do:
regress toward the smooth conditional mean.

Beyond k ≈ 48 the source's ratio rises steeply and reaches ~7 × 10¹⁰ at k = 63. This is a
numerical edge effect — its reference spectrum carries negligible energy at the Nyquist boundary,
so the ratio is unstable — and it should be excluded rather than averaged into a band. (An
earlier version of this analysis grouped k = 33–63 into a single band, which allowed that
instability to dominate the average and appeared to show the port closer to truth at fine
scales. It is not.)

Note also where each implementation's spectral *error* is located: 86.8% of this port's total
lies in k ≤ 4 and 70.5% of the source's in k = 5–16, in both cases bands where the energy ratio
is within 0.4% of unity. Because e<sub>spec</sub> normalises by total energy, and energy resides
at large scales, the metric is dominated by sub-percent deviations at the largest eddies rather
than by fine-scale fidelity. The 3.5-fold difference should be read with that in mind.

**No claim is made here about the backbone.** The two implementations differ simultaneously in
architecture (ConvNeXt against this repository's UNet), inference scheme (patch-based against
full-field), resolution, input degradation, training budget (4000 against 2000 epochs) and
exponential moving average (used by the source, absent here — and Section 5.9 measures that its
absence alone produces ±35% run-to-run variation in precisely the high-wavenumber band). Four of
these are independently known to affect spectral fidelity, so the contribution of the backbone
cannot be isolated from this experiment. A matched-backbone study is left to future work.

**The field-error rows must not be read as this port outperforming the source.** The source's
reconstruction is *less* pointwise-accurate than its own input (0.464 → 0.591, correlation
0.8864 → 0.8262). This is expected rather than a deficiency: the source treats the target as a
draw from a conditional distribution — its own figures label the reference `x1 ~ p(x1|x0)` — and
a low-pass filter leaves large conditional entropy, so the method synthesises small-scale
structure that is statistically correct and pointwise uncorrelated with any particular
realisation. A near-perfect spectrum accompanied by falling correlation is exactly that
signature.

**This explains the choice of metrics in each setting, and it qualifies the headline result of
this chapter.** Sparse point sensors constrain the field far more tightly than a low-pass
filter: 1024 exact values collapse the conditional entropy to the point where a single
realisation is nearly determined, which is what makes pointwise evaluation meaningful here and
distributional evaluation (KL, W1, dissipation) the appropriate choice in `[Schiodt2026]`.
The result in Table 5.1 should therefore be read not as "stochastic interpolants are more
accurate than diffusion in general", but as: *the interpolant formulation, applied to a
measurement-preserving task of low conditional entropy, delivers pointwise accuracy that the
same formulation does not target on a low-pass task.*

**A metric that does not transfer.** The source's headline diagnostic — dissipation recovered
relative to truth, 0.724 → 1.012 — is vacuous on this benchmark. The Voronoi-filled sensor
input already sits at 0.998 of the true enstrophy, because the block discontinuities of
nearest-neighbour interpolation supply variance equal to the missing fine-scale variance for
entirely the wrong reason. There is no deficit for a method to close. Readers arriving from
`[Schiodt2026]` will reach for this metric first, and it must be reported as inapplicable rather
than as satisfied.

Figures: `paper_repro_plots/spectrum_ensemble_avg.png`, `vorticity_field.png`.

`[TODO: the authors' shipped pipeline required three fixes to run end to end — an import path
assumption, a working-directory assumption, and an inconsistent sampler-step constant between
the simulation and analysis scripts. Worth one sentence if you want to make a reproducibility
observation; omit if it reads as ungenerous.]`

---

## 5.10 Synthesis

The comparison resolves into a single question: **how much of the measurement is allowed to
survive into generation?**

Methods that destroy it — the diffusion baseline, and DPS, both noising to t = 400 — obtain
near-perfect immunity to sensor error (2.4% and 2.9% pass-through) and pay for it in accuracy,
losing 16% of the field variance and a factor of 2.5 in MSE. The method that preserves it makes
the opposite trade: it passes sensor error through essentially one-to-one, and it wins
decisively at any noise level a real instrument would produce. The crossover lies at a 31%
sensor error against a realistic 1–5%.

That trade is the contribution. It is not that stochastic interpolants are better in general —
this chapter has documented three respects in which they are not. They produce the wrong
cascade slope (Δs 0.39, the worst of the three). They lose the spectral comparison to DPS
entirely once the data distribution shifts, by a factor of 28 at the same nominal Reynolds
number. And their advantage in field accuracy, while it survives a twenty-fold range of
Reynolds number, erodes steadily across it from 27% to 9.5%.

What generalises is narrower and more useful than a ranking. Preserving the measurement is
worth roughly a factor of two in field accuracy on this benchmark, at the price of proportional
sensitivity to measurement error — a price worth paying whenever instruments are accurate to
better than about 30%. And the limiting constraint on both measurement-preserving approaches is
not accuracy but simulability: SI and DPS both require that the measurement operator can be
modelled, while the diffusion baseline requires no model of it at all. Where the real
degradation cannot be faithfully simulated — genuine PIV optics, particle dynamics, correlated
instrument noise — that is the constraint that will bind, and it binds SI and DPS equally.

`[TODO: one or two sentences connecting to your conclusions chapter / future work.]`

---

## Appendix — artifact index

| section | figures and tables |
|---|---|
| 5.2 | `metrics_table.txt`, `metrics_table_jcp.md`, `stats_baseline_dps_si_vs_reference.png`, `spectrum_ratio_3runs.png`, `energy_map_{kinetic,enstrophy}_*.png` |
| 5.3 | `noise_passthrough.png` |
| 5.4 | physics ablation, r-sweep, sample_step sweep, t-sweep, λ-sweep (all in `experiments/kmflow_re1000_rs256_ddim_conditional_new/`) |
| 5.5 | `residual_vs_spectrum.png` |
| 5.6 | `si_robustness_curve.png`, `degradation_families.png` |
| 5.7 | `reanalysis_divergence.png` |
| 5.8 | `cross_re_trend.png` |
| 5.9 | `baseline_reproduction.png`, `stats_baseline_repro_vs_reference.png`, `paper_repro_plots/*` |

Reproduction commands for every figure are in the scripts of the same name at the repository
root; `SI_README.md` §10 documents the deviations from `[Schiodt2026]`.
