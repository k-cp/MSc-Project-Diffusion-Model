# Inpainting literature map

_Working notes, verified against the PDFs in `~/Desktop/MSC projet/Impainting/`._
_Source of truth: `memory/inpainting-literature.md`. Edit there, not here._


Searched 2026-08-06 for the [[inpainting-dead-block]] writeup. **The Rome Tor Vergata group
(Biferale / Buzzicotti / Li / Bonaccorso / Lanotte) owns this problem** — go to them first.

########## ZHENG ET AL. VERIFIED FROM THE PDF 2026-08-12 — CORRECTS THIS FILE ##########
PDF IS LOCAL: ~/Desktop/MSC projet/Impainting/mask_flow_reconstruction.pdf (34 pp).
Zheng, Li, Ma, Fu & Li, Phys. Rev. Fluids 9, 024608 (2024). Editors' Suggestion. NO arXiv,
paywalled — the PDF on the Desktop is the only access.
**SETUP (kills any numeric comparison):** compressible ISOTROPIC turbulence, 2D slices of a
512^3 DNS (so 512^2 slices), reconstructing **DENSITY**, 12,480 train / 1,500 val / 1,500
test slices. Gap ratios 39.06% and 56.25% only. Metric = normalised **MAE (L1)**:
MAE*_d = MAE_d/rho_rms, MAE*_g = MAE_g/grad-rho_rms, damaged region only (their Eq 6/7).
Their grad is dp/dx + dp/dy (a SUM); our mae_g_in uses the MAGNITUDE — deliberate, documented.
**SDF/MDF/RDF ARE THEIR TERMS — verified p.9. Fig 1(c) IS our three families — verified.
"192/96/8 px at 512^2" IS CORRECT — verified (SDF 384^2, MDF 4x192^2, RDF 16x16 cells).**
**THE NUMBERS IN void_masks.py ARE WRONG.** Table V (PCGAN^0%) is 0.606 / 0.480 / 0.035,
not 0.615/0.477/0.034. Ratio 17.3x, not 18x.
**THE BIG ONE — TWO TABLES, AND ONLY ONE IS COMPARABLE:**
   Table I  PCGAN^100% (headline)  SDF 0.363  MDF 0.294  RDF 0.028
   Table V  PCGAN^0%              SDF 0.606  MDF 0.480  RDF 0.035
   Their headline model receives COMPLETE VELOCITY FIELDS (u,v) as extra input ("physical
   constraints") — it is NOT inpainting from the damaged field alone. 0.606->0.363 is a
   40.1% gain from that auxiliary data. **PCGAN^0% is the like-for-like column for us.**
   Their DCGAN baseline (Table II, RDF 56.25%): MAE*_d 0.246, MAE*_g 0.810.
**THEIR OBJECTIVE IS ALSO PERCEPTUAL + ADVERSARIAL — verified 2026-09-01 from the PDF.**
They "employ the perceptual loss function, which evaluates the difference between features
extracted", and they "highlight the importance of the PatchGAN discriminator and the
perceptual loss function to achieve accurate reconstruction". So BOTH benchmark papers and
the VQ-GAN source use the same two terms, which is why perc, gan and p+g are the arms of
our ablation and why p+g is the one with a published counterpart. Cited in sec:m-aux as the
fluids precedent for the perceptual term (Johnson et al. is the natural-images origin).

**TWO PRECEDENTS WE SHOULD CITE:**
 1. THEY MEASURED OUR DEPTH CURVE. p.11-12 + Fig 5: MAE*_elem,d "closely related to the
    distance from the boundary... the closer to the center, the larger", following an
    "approximately LINEAR trend with respect to the distance from the boundary". Their p.10
    mechanism sentence is our finding verbatim. AND: MAE*_elem,g (gradient) does NOT show
    that dependence — "relatively constant within the core". TESTABLE on our mae_g_in.
 2. THEY USE OUR TRAINING PROTOCOL. p.4-5, for the RDF mask: "during network training, the
    location of the damaged regions is randomly altered... However, for the validation and
    testing datasets, we deliberately keep the position of damaged regions fixed."
    Train-on-re-rolled -> test-on-fixed has a published precedent; stop framing it as a
    self-imposed handicap.
NUMERIC COMPARISON REMAINS INVALID: different flow, different variable (density vs
vorticity), different resolution, different metric (L1 vs L2 ratio), zero coverage overlap.
Cite qualitatively only.

########## SHU ET AL. VERIFIED 2026-08-12 — CORRECTIONS ##########
Table 1 reproduces verbatim; data spec matches ours on every element (Kolmogorov Re=1000,
vorticity, 256^2, 40 runs from 2048^2, 36/4, dt=1/32, f=-4cos(4x2)-0.1w).
 * FOUR authors: Shu, **Zhen, Li**, Barati Farimani. Not "Shu & Barati Farimani".
 * It is a VQ-VAE trained with the **VQ-GAN objective** (Esser et al.) — the discriminator
   is in the loss, not the generator family. Do not call it "a GAN".
 * **THE OBJECTIVE IS PERCEPTUAL *AND* ADVERSARIAL, NOT ADVERSARIAL ALONE — verified
   2026-09-01 from p.9.** Quotable: they "adopt the training strategy for VQ-GAN which
   combines the standard VQ-VAE loss function with GAN loss and perceptual loss", and the
   Stage-1 objective is written `min max E[L_VQ + L_percept + lambda*L_GAN]`. Their
   perceptual term "is computed using a pretrained VGG network to monitor the
   reconstruction error at multiple feature map levels" — i.e. THE SAME CONSTRUCTION AS
   OUR perc TERM (frozen ImageNet-VGG16 feature maps).
   CONSEQUENCE: our **p+g arm is their objective**, not a fourth arm of our own devising,
   and `gan` alone is only half of it. The thesis said the discriminator was "the one
   ingredient separating the benchmark's objective from plain regression" until 2026-09-01;
   that was wrong and is now corrected in sec:m-aux and sec:inp-aux.
   CAVEAT: the quoted objective is **Stage 1**. Stage 2 fine-tunes for completion with the
   decoder frozen, and which terms stay active there is NOT verified. Say "their VQ-GAN
   training strategy combines reconstruction, perceptual and GAN losses", not "at every
   stage".
   TRAP THAT CAUSED THE ERROR: grepping their PDF for `johnson2016perceptual` finds it, but
   in a RELATED-WORK list of image-inpainting techniques "potentially applicable" to flow
   completion (p.5). A citation there is not evidence of use. Only the Method section
   settles it.
 * "The location of the masks are fixed throughout model training and inference" — VERBATIM,
   quote this. But "a separate model per mask configuration" is NOT STATED anywhere; drop it.
 * "25%" is never printed — it is exact arithmetic from their stated mask sizes. Attribute
   as derived.
 * **Eq 6's AGGREGATION over the test set is never stated** (spatial indices only, no sample
   index). Our pooled ratio-of-sums is one reading; per-frame mean is another. FIX: report
   both. If they bracket 0.6533/0.3594, "our number spans Shu's under both defensible
   readings" is stronger and safer than any parity claim.
 * No code or data released. Ours is an INDEPENDENT REALISATION of the same spec (1272
   frames vs their ~1280), not "the same data".
 * FactFormer is THEIR OWN GROUP's prior work (Li, Shu & Barati Farimani, NeurIPS 2023,
   arXiv:2305.17560). FNO is Li et al. 2021. Both trained by them from scratch; no classical
   or null baseline anywhere in the paper.

=== THE DIRECT COMPARABLE — READ THIS ONE ===
**Li, Lanotte, Buzzicotti, Bonaccorso, Biferale, "Multi-scale Reconstruction of Turbulent
Rotating Flows with Generative Diffusion Models", arXiv:2312.11121 (Atmosphere 2024).**
Same experiment as ours, different flow. RePaint vs Palette vs GAN on a CENTRED SQUARE GAP.
- DATA: TURB-Rot DNS, rotating turbulence, Ro~0.1, 256^3 -> Galerkin-truncated to 64^3.
  2D horizontal slices, **velocity MAGNITUDE** (not components, not vorticity).
  84,480 train / 20,480 test, augmented by random periodic shifts from 600+160 snapshots.
  Train/test separated by >3400 integral times.
- GAPS: l/l0 = 24/64, 40/64, 62/64 (0.375 / 0.625 / 0.969 of the domain). Integral scale
  L ~ 0.15 L0, i.e. ~9.6 px, so their gaps are ~2.5 / 4.2 / 6.5 INTEGRAL SCALES across —
  the same regime as our 64x64 = 4.6 correlation lengths.
- TRAINING: Palette = conditional, paired, **a SEPARATE model per mask size**; measurement
  region frozen, forward process only inside the gap. RePaint = unconditional DDPM + the
  resample/jump trick, no mask-specific training. GAN = MSE + adversarial, deterministic.
- METRICS: normalised MSE in the gap (÷ sigma_pred*sigma_true), **Jensen-Shannon divergence**
  of PDFs (field and gradients), E(k), 4th-order flatness of increments.
- RESULT: **Palette (conditional/paired) > RePaint (guided unconditional) > GAN** on MSE;
  RePaint best on gradient statistics. "When an entire vortex structure is missing... all
  models fail significantly."
**WHY IT MATTERS TO US: their taxonomy maps onto our three methods almost exactly** —
SI = Palette-like (conditional, paired, starts at the measurement), JCP baseline / DPS =
RePaint-like (guided unconditional prior). Their ranking REPRODUCES ours. Strong external
corroboration for the SI-wins result, on a different flow and a different code base.

=== THE ANCESTOR ===
Buzzicotti, Bonaccorso, Clark Di Leoni, Biferale, "Reconstruction of turbulent data with deep
generative models for semantic inpainting from TURB-Rot database", **Phys. Rev. Fluids 6,
050503 (2021)**, arXiv:2006.09179. ~300K 2D turbulent images, public database. Two
Context-Encoder approaches: (a) L2 pixelwise + small adversarial penalty, (b) search the
closest encoding of the corrupted field in a pretrained generator's latent space. Compares
against **Nudging** (data assimilation) — a non-ML baseline worth citing.

=== SAME BENCHMARK AS OURS (Kolmogorov Re=1000, 256x256!) ===
**"Guiding diffusion models to reconstruct flow fields from sparse data", arXiv:2510.19971
(Oct 2025).** Introduces "masked diffusion" (smooth Gaussian mask in the reverse process,
scheduler gamma_t=(t/T)^3) as an alternative to DPS. 2D Kolmogorov Re=1000 at 256x256,
3000 train / 1240 test; also 3D HIT from JHTDB. Unconditional DDPM, ~100 DDIM steps.
**Sparsity levels 5% / 1.5625% / 0.1% — 1.5625% IS EXACTLY our 1024/65536 u3232.**
Benchmarks explicitly against **Shu et al. (our JCP baseline)**: RMSE 0.13 vs 0.18, spectrum
error 0.26 vs 0.54 at 5%; 7.7 s/sample vs 25 s. Scattered points only, NO contiguous holes.
=> Closest published competitor to our setup. Cite it; also note it does NOT do dead blocks.

**Shu & Barati Farimani, "Inpainting Computational Fluid Dynamics with Deep Learning",
arXiv:2402.17185. THIS IS "SHU ET AL." IN THE INPAINTING CHAPTER — the benchmark
[[si-inpainting-fork]] scores against. PDF: `~/Desktop/MSC projet/Impainting/`.**
**NAMING TRAP: there are TWO "Shu et al." in this project.** Same first author, different
papers: the JCP 2023 super-resolution diffusion model is the SR chapter's baseline; THIS one
is the VQ-VAE inpainting benchmark. "Shu et al. Table 1" always means this one.
Kolmogorov Re=1000, VQ-VAE-style two-stage discrete latent model. **Data is IDENTICAL to ours**
(read from the paper 2026-08-11): 40 runs, 36 train / 4 test, 2048^2 downsampled to 256^2,
dt = 1/32 s, pseudo-spectral solver of Li et al.
**TABLE 1 HAS THREE ROWS AND EVERY ONE IS 25% COVERAGE** — they vary ARRANGEMENT, not amount:
  16 x 32^2 (4 rows x 4 cols)  0.1663 / FNO 0.5175 / FactFormer 0.3374
   4 x 64^2 (evenly spread)    0.3594 / FNO 0.7321 / FactFormer 0.7044
   1 x 128^2 (centred)         0.6533 / FNO 0.9278 / FactFormer 0.7134
Each removes 16384 of 65536 px. So matching them needs the 16-void run, NOT other gap rates.
Their metric is Eq 6, normalised by the truth's energy inside the mask. Their own conclusion:
models are "more vulnerable to extended continuous regions of missing data" — our depth result
without the correlation-length normalisation, which is the gap we fill.
They train a separate model per mask configuration on a FIXED mask (memorisable) — hence the
`center` controls. Figures 4-6 are Samples #1-3; #1 is the stripy laminar field.

=== LAGRANGIAN / TEMPORAL GAPS ===
Li, Biferale, Bonaccorso, Buzzicotti, Centurioni, "Stochastic reconstruction of gappy
Lagrangian turbulent signals by conditional diffusion models", **Comm. Physics (2025)**,
arXiv:2410.23971. **Expresses gap length in Kolmogorov times: tau_eta to ~100 tau_eta** —
the temporal analogue of our correlation-length normalisation, and the precedent for it.
327,680 DNS trajectories (90/10); REAL DATA: 19,396 NOAA Global Drifter Program trajectories
-> 116,486 60-day segments. U-Net, 800 diffusion steps, batch 256, 4x A100, ~24 h.
Baseline = Gaussian Process Regression. C-DM's edge GROWS with gap size; captures
acceleration excursions to 40 sigma that GPR cannot.

=== OTHER FLUID GAP-FILLING (context, not direct comparables) ===
- Ensemble flow reconstruction in the atmospheric boundary layer via LATENT diffusion,
  Phys. Fluids 35, 126604 (2023), arXiv:2303.00836 — reconstructs from <1% volume coverage.
- "Vector-based loss functions for turbulent flow field inpainting", arXiv:2509.05787 —
  U-Net on EXPERIMENTAL PIV (TCC-III engine); cosine-similarity + magnitude losses beat
  per-component MSE. The practical/experimental end of the field.
- Oommen, Khodakarami, Bora, Wang, Karniadakis, "Learning turbulent flows with generative
  models for super resolution and sparse flow reconstruction", **Nature Comms (2026)**,
  arXiv:2509.08752, code Gen4Turbulence — adversarial neural operator, 114x faster than
  diffusion forecasters.
- Buzzicotti et al., "GANs to infer velocity components in rotating turbulent flows",
  Eur. Phys. J. E (2023).
- Classical baseline to cite: **gappy POD** (Everson & Sirovich) — needs a POD basis from
  complete data. Also CNN/CVAE gap-filling for PIV (Zhang 2022; VAE for reacting PIV
  arXiv:2312.06461).

=== THE CV CANON THEY ALL INHERIT FROM ===
- **Context Encoders** (Pathak et al., CVPR 2016) — central 64x64 hole in 128x128, L2 +
  adversarial. The template Buzzicotti 2021 ported.
- **RePaint** (Lugmayr et al., CVPR 2022, arXiv:2201.09865) — pretrained UNCONDITIONAL DDPM,
  never trained on inpainting; conditions by resampling the known region each reverse step,
  plus the jump-back-forward "resampling" trick to fix semantic inconsistency. Mask-agnostic.
  CelebA-HQ / ImageNet, masks: wide, narrow, super-resolve, expand, half.
- **Palette** (Saharia et al., SIGGRAPH 2022, arXiv:2111.05826) — CONDITIONAL image-to-image
  diffusion, trained on ImageNet (+Places2). Masks: 10-20 / 20-30 / 30-40% free-form and a
  128x128 CENTRE RECTANGLE.
- **LaMa** (Suvorov et al., WACV 2022, arXiv:2109.07161) — fast Fourier convolutions for a
  global receptive field + large training masks. The "large hole" specialist.
KNOWN CV CONSENSUS, matches our result: methods tuned for SMALL holes (LaMa, Palette) degrade
on large ones; large-hole inpainting produces PLAUSIBLE, not CORRECT, content.

=== THE GAP OUR EXPERIMENT FILLS (checked: nobody does this) ===
No fluid inpainting paper normalises hole size by a MEASURED correlation length, and none
reports **skill vs a null fill as a function of depth into the hole**. Li et al. give the
integral scale but quote gaps as a domain fraction; the Lagrangian paper uses tau_eta but is
1D-in-time. So our 14 px / 4.6-correlation-lengths / zero-skill-past-4px analysis is a
methodological contribution, not just a negative result. Also nobody combines the hole with
sparse sensors elsewhere — our dead block sits inside a 1.56%-coverage field, theirs sit in
fully-observed fields.
