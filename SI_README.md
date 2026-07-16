# Stochastic Interpolant Super-Resolution — Training & Inference Guide

Reproduction guide for the Stochastic Interpolant (SI) super-resolution feature added to
this repository, adapting **Schiødt, Mücke & Velte, _"Generative super-resolution of
turbulent flows via stochastic interpolants"_, Scientific Reports 16:4229 (2026)** to this
project's Kolmogorov-flow vorticity data.

This document is written so that someone with **only this file and `si_ckpt.pth`** can run
inference. Everything needed is stated explicitly.

> **The single most important thing to know:** SI is **not** a toggle on the diffusion
> model. It is a **separate network trained from scratch** and shares no weights with
> `conditional_ckpt_new.pth`. You cannot run SI without a trained `si_ckpt.pth`.

---

## 0. TL;DR

```bash
# Train (once, ~15 h on one GPU) -> writes pretrained_weights/si_ckpt.pth
sbatch run_train_si.sh

# Infer with the trained checkpoint (~15-25 min)
sbatch run_inference_conditional.sh si

# Results land in:
#   experiments/kmflow_re1000_rs256_ddim_conditional_new/
#       si_guided_recons_u3232_t400_r20_w0.0/sample_batch{0..63}/
```

---

## 1. What you need

### 1.1 Files

| File | Size | Required for | Notes |
|---|---|---|---|
| `pretrained_weights/si_ckpt.pth` | 40 MB | inference | the trained drift network |
| `configs/kmflow_re1000_rs256_conditional.yml` | 1 KB | both | **must match training** — rebuilds the architecture |
| `data/kmflow_sampled_data_irregnew.npz` | 6.7 GB | both | the low-res input `x₀` |
| `data/kf_2d_re1000_256_40seed.npy` | 3.4 GB | both | ground truth — **also needed for the scaler**, see §6.3 |
| `main.py` | — | inference | entry point / routing |
| `train_si.py` | — | training | training loop |
| `runners/stochastic_interpolant.py` | — | both | SI core + runner |
| `runners/rs256_guided_diffusion.py` | — | both | data loaders, `StdScaler`, metrics |
| `models/diffusion_new.py` | — | both | the UNet (`ConditionalModel`) |
| `functions/` | — | both | imported by the loaders |

### 1.2 Environment

```bash
module load cray-python/3.11.7                       # Isambard
source /path/to/diffusion_env/bin/activate
export ATEN_CPU_CAPABILITY=default
export USE_MKLDNN=0
```

Python 3.11. Packages:

```
torch          # tested with the cluster's CUDA build
numpy
einops         # slice2sequence
tqdm
PyYAML
torchvision    # imported by rs256_guided_diffusion
matplotlib     # image grids
scipy          # ONLY for metrics.py (gaussian_kde)
```

### 1.3 Hardware

- **1 GPU.** The code is single-GPU: there is **no** `DataParallel`/`DDP`, so requesting
  `--gpus=2` leaves the second GPU idle.
- Training ≈ **15 h** at 2000 epochs. Isambard's **max walltime is 24 h** — see §5.6 for resume.
- CPU-only works but is impractically slow.

---

## 2. The data

Both arrays are `(40, 320, 256, 256)` = 40 independent simulations × 320 time frames ×
256×256 grid.

| | File / key | dtype | What it is |
|---|---|---|---|
| **x₁** (target) | `kf_2d_re1000_256_40seed.npy` | float32 | high-res **vorticity** ground truth. mean ≈ 0, std ≈ 4.79, range ≈ ±20 |
| **x₀** (base) | `kmflow_sampled_data_irregnew.npz` → `u3232` | float64 | the low-res observation |
| sensor map | `kmflow_sampled_data_irregnew.npz` → `idx_lst` | int64 | `(40, 1024)` flat indices into the 256×256 grid |

**What `u3232` actually is:** exactly **1024 unique values per frame** — the sensor
readings, **nearest-neighbour (Voronoi) filled** across the grid. It is *not* a blur or a
downsample. `u3232 == ground truth` **exactly** at the `idx_lst` locations (verified: error
= 0.0); everything between sensors is fabricated by the fill.

- **Coverage: 1024 / 65536 = 1.56 %** of the grid is observed.
- Each trajectory has its **own random sensor layout**, reused for all 320 of its frames.
- The name `u3232` refers to the *density* of a 32×32 grid (1024 = 32²), but the points are
  irregularly scattered.

> **Naming warning:** the code calls this `blur_data` (from `load_recons_data`). **Nothing
> is blurred.** There *is* an optional Gaussian blur in `load_recons_data`, but it is gated
> by `config.data.smoothing`, which is `False`. Do not describe `x₀` as "blurred" or
> "filtered" — it is a sparse-sensor field.

### 2.1 Splits

| Split | Trajectories | Used for |
|---|---|---|
| train | `[:-4]` — first **36** | SI training |
| test | `[-4:]` — last **4** | inference / evaluation (**held out**, no leakage) |

Samples are **3-frame stacks**: for trajectory `i`, frame `j` → `x[i, j:j+3]` → `(3, 256, 256)`.
**The 3 channels are 3 consecutive time frames**, not velocity components.

---

## 3. The checkpoint

`pretrained_weights/si_ckpt.pth` (40 MB) is a `torch.save` of:

```python
{"epoch": <int>, "model": <state_dict>, "optimizer": <AdamW state_dict>}
```

- `model` — the trained drift network weights. This is all inference needs.
- `epoch` / `optimizer` — only used by `--resume`.

The loader accepts three formats for backward compatibility:

```python
states = torch.load(ckpt_path, map_location=device)
if isinstance(states, dict) and "model" in states:
    state_dict = states["model"]        # current format
elif isinstance(states, (list, tuple)):
    state_dict = states[-1]             # legacy [state_dict]
else:
    state_dict = states                 # raw state_dict
```

**The checkpoint contains weights only.** It does **not** contain the architecture (rebuilt
from `--config`) or the normalization statistics (see §6.3). `*.pth` is gitignored, so git
is **not** a backup — that 40 MB file is ~15 h of GPU time.

---

## 4. The model

The drift network `b_θ` **reuses this repo's `ConditionalModel` UNet**, trained from scratch
(random init). Architecture from `kmflow_re1000_rs256_conditional.yml`:

```yaml
model:
    type: "conditional"
    in_channels: 3      # 3 consecutive frames
    out_ch: 3
    ch: 64
    ch_mult: [1, 1, 1, 2]
    num_res_blocks: 1
    attn_resolutions: [16]
    dropout: 0.0
data:
    image_size: 256     # hard-asserted at runtime
diffusion:
    num_diffusion_timesteps: 1000   # reused ONLY as the pseudo-time scale (see below)
```

The UNet's three inputs map onto SI as:

| UNet arg | SI meaning |
|---|---|
| `x` | `I_τ` — the interpolant (bridge state), `(B,3,256,256)` |
| `t` | pseudo-time `τ`, **scaled ×1000** |
| `dx` | `x₀` — the conditional low-res field, `(B,3,256,256)` |
| output | the predicted **drift**, `(B,3,256,256)` |

`x₀` passes through `emb_conv` (1×1 conv → GELU → 3×3 **circular** conv — circular because
the domain is periodic), is concatenated with `conv_in(I_τ)`, and fused by `combine_conv`.

**The `×1000` scaling** (`time_scale = config.diffusion.num_diffusion_timesteps`): the UNet's
sinusoidal `get_timestep_embedding` was designed for diffusion timesteps 0–1000. Feeding raw
`τ ∈ [0,1]` would leave the embedding nearly constant across the whole interval. The constant
is arbitrary but **must be identical in training and inference** — it is, both go through
`StochasticInterpolant.drift()`.

---

## 5. Training — exactly how it was done

### 5.1 Command

```bash
sbatch run_train_si.sh                 # defaults: 2000 epochs, frame_stride 4
sbatch run_train_si.sh 4000 2          # EPOCHS FRAME_STRIDE
sbatch run_train_si.sh 2000 4 resume   # resume from si_ckpt.pth
```

which runs:

```bash
python train_si.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --epochs 2000 \
    --batch_size 32 \
    --lr 2e-4 \
    --frame_stride 4 \
    --resume 0 \
    --si_ckpt ./pretrained_weights/si_ckpt.pth
```

### 5.2 Hyperparameters (exact)

| Parameter | Value |
|---|---|
| epochs | 2000 |
| batch_size | 32 |
| optimizer | AdamW |
| lr | 2e-4 |
| weight_decay | 1e-4 |
| schedule | linear warmup 50 epochs → cosine anneal to 0 |
| warmup_epochs | 50 |
| frame_stride | 4 |
| num_workers | 0 |
| save_every | 100 epochs |
| seed | 1234 |
| σ noise scale | 0.1 |

### 5.3 Data preparation

```
train split = trajectories [:-4]                    -> 36
3-frame stacks: for j in range(0, 320-2, frame_stride=4)   -> 80 per trajectory
total training pairs = 36 × 80                      -> 2880
batches per epoch = 2880 / 32                       -> 90
```

`frame_stride=4` subsamples frames to decorrelate samples (the paper uses temporally
decorrelated snapshots) and to keep memory sane. Standardization is applied **per batch**,
not to the whole tensor, to avoid a second full-size copy.

### 5.4 The objective

Per training step, for each sample in the batch:

```
τ ~ U[0, 1]                               shape (B,1,1,1)
z ~ N(0, I)                               shape (B,3,256,256)
W_τ = √τ · z                              Wiener process: W_τ ~ N(0, τ), W_0 = 0

α_τ = 1 − τ        α̇_τ = −1
β_τ = τ²           β̇_τ = 2τ
σ_τ = 0.1(1 − τ)   σ̇_τ = −0.1

I_τ = α_τ·x₀ + β_τ·x₁ + σ_τ·W_τ           the interpolant   (paper Eq 5)
R_τ = α̇_τ·x₀ + β̇_τ·x₁ + σ̇_τ·W_τ           its τ-derivative  (paper Eq 9)

pred = b_θ(I_τ, τ×1000, x₀)
loss = mean( (pred − R_τ)² )              drift matching     (paper Eq 8)
```

**Critical:** `I_τ` and `R_τ` are built from the **same `z`**. `R_τ` is the analytic
derivative of *that specific* bridge realization.

**Boundary conditions** (paper Eq 6) — these are what make the bridge valid, and they hold
exactly: at τ=0, `α=1, β=0, W₀=0` ⟹ **I₀ = x₀**. At τ=1, `α=0, β=1, σ=0` ⟹ **I₁ = x₁**.

**Why regress onto `R_τ`?** At inference `x₁` is unknown, so `R_τ` can't be computed. The
minimizer of `E‖b − R‖²` is the conditional expectation `E[R_τ | I_τ, x₀]` — the average
bridge velocity over all plausible `x₁`, which is exactly the drift that transports the
distribution correctly.

**Deviation from the paper:** Eq 9 sums over a *fixed grid* of `N_τ` pseudo-times × all
samples. This implementation instead draws **one random τ per sample per step**. Both are
unbiased estimators of the same integral `∫₀¹ E[·] dτ`; random-τ is the standard
diffusion/flow-matching practice and far cheaper per step.

**Loss magnitude:** the code uses `((pred - target)**2).mean()` — averaged over batch,
channels **and all 65,536 pixels**. The paper's `‖·‖²` is a sum over the field. The two
differ by a constant ≈ 3·256·256 ≈ 196,000, which only rescales the gradient (absorbed by
the lr). **Do not compare the reported loss to any number in the paper.**

### 5.5 Runtime & expected log

~**27 s/epoch** → 2000 epochs ≈ **15 h**. Log format:

```
2026-... - Loading training pairs (x0=low-res, x1=high-res)...
2026-... - Train pairs: 2880 (frame_stride=4), mean=-0.0000 std=4.7870
2026-... - epoch 1/2000     loss=2.4xxxxx   lr=4.00e-06
...
2026-... - epoch 2000/2000  loss=0.00xxxx   lr=...
2026-... - Saved checkpoint (epoch 2000) -> ./pretrained_weights/si_ckpt.pth
2026-... - Training finished.
```

**Health check:** `loss` must trend **down**; `lr` **rises then falls** (warmup → cosine) —
that is expected, not a bug. A 100-epoch smoke run reached `loss ≈ 0.0047`.

### 5.6 Resume (required to train past 24 h)

Isambard's walltime cap is **24 h and cannot be raised**. Checkpoints are written every 100
epochs, so a timeout loses at most 100 epochs. To continue:

```bash
sbatch run_train_si.sh 2000 4 resume
```

This loads model **and** optimizer **and** epoch, and continues. Confirm in the log:

```
Resumed from ./pretrained_weights/si_ckpt.pth at epoch <N>
```

Use the **same** epochs/frame_stride as the original run so the LR schedule stays consistent.
`resume` is for continuing an *interrupted* run; to train *additional* epochs past a completed
run, raise `--epochs` and pass `resume`.

---

## 6. Inference from the checkpoint

### 6.1 Command

```bash
sbatch run_inference_conditional.sh si
```

or directly:

```bash
python main.py \
    --config kmflow_re1000_rs256_conditional.yml \
    --seed 1234 \
    --run_si 1 \
    --si_ckpt ./pretrained_weights/si_ckpt.pth \
    --si_steps 100
```

| Flag | Default | Effect |
|---|---|---|
| `--run_si 1` | 0 | routes to `SIRunner` (**takes precedence over `--run_dps`**) |
| `--si_ckpt` | `./pretrained_weights/si_ckpt.pth` | which trained model to load |
| `--si_steps` | 100 | SDE integration steps; 50 ≈ halves runtime |
| `--seed` | 1234 | changes sampling noise → a **different but equally valid** sample |
| `--repeat_run` | 1 | N samples per batch → `sample_run_0_it0`, `sample_run_1_it0`, … |

`--t`, `--r`, `--zeta`, `--operator`, `--scale_factor` are **ignored** by SI. `--t`/`--r`
still appear in the output folder name only because the naming code is shared.

### 6.2 What happens internally

```
main.py  --run_si 1
   ├─ dir_name = "si_" + "guided_recons_u3232_t400_r20_w0.0"
   └─ SIRunner(args, config, logger, log_dir)
        ├─ StochasticInterpolant(config, device)   # fresh ConditionalModel
        ├─ torch.load(si_ckpt) -> load_state_dict  # trained weights
        ├─ model.eval()
        └─ si_sample_pipeline()
             load_recons_data -> ref_data, blur_data, mean, std
             scaler = StdScaler(mean, std)
             for each of 64 batches (batch_size=20):
                 x₀ = scaler(blur)
                 X  = si.sample(x₀, n_steps=100)     # Heun SDE
                 sample = scaler.inverse(X)
                 log L2 + vorticity residual; save png + npy
```

The sampler (paper Eq 7), a **stochastic Heun** predictor–corrector:

```python
X = x0                                        # X_{τ=0} = x0  (NOT noise)
for τ in linspace(0, 1, n_steps+1)[:-1]:
    dτ    = 1 / n_steps
    σ_τ   = 0.1 * (1 - τ)
    noise = σ_τ * sqrt(dτ) * randn_like(X)    # dW_τ = √dτ·z  (note √dτ, not dτ)

    b1 = b_θ(X,  τ,    x0)                    # predictor
    Xp = X + b1*dτ + noise
    b2 = b_θ(Xp, τ+dτ, x0)                    # corrector
    X  = X + 0.5*(b1+b2)*dτ + noise           # same noise in both stages
```

`σ_τ → 0` as `τ → 1`, so the final steps are effectively deterministic.

### 6.3 ⚠️ The scaler is NOT in the checkpoint

The network only ever saw **standardized** data. `mean`/`std` are **recomputed at runtime**
from the ground-truth file and are **not** saved with the weights. Get them wrong and the
output is garbage.

They are the mean/std of `kf_2d_re1000_256_40seed.npy[:-4]` (the 36 training trajectories,
754,974,720 values):

```
mean = -2.2565286572392790e-08     # i.e. 0 to numerical precision
std  =  4.786953449708109
```

**This makes the checkpoint portable.** If you do not have the 3.4 GB ground-truth file, you
can hardcode:

```python
scaler = StdScaler(0.0, 4.786953449708109)
```

and run inference with only `si_ckpt.pth` + your own `x₀`. (The ground truth is otherwise
only needed to score L2 / residual.)

### 6.4 Output layout

```
experiments/kmflow_re1000_rs256_ddim_conditional_new/
└── si_guided_recons_u3232_t400_r20_w0.0/
    ├── config.yml
    ├── logging_info.txt
    └── sample_batch{0..63}/
        ├── input_image.png      input_arr.npy           # x₀ (sensor field)
        ├── reference_image.png  reference_arr.npy       # ground truth
        └── sample_run_0_it0.png sample_arr_run_0_it0.npy # the reconstruction
```

Each `sample_batch<i>` is **wiped before writing**, so a folder still containing
`metric_log_*.pkl` means the SI run never wrote there. `.npy` files are `(20, 256, 256)` —
the **middle frame** of each 3-frame stack (`slice2sequence`). Filenames deliberately match
`reconstruct()` so `animate_results.py` and `metrics.py` read them unchanged.

### 6.5 Runtime

~**15–25 min** for all 64 batches: 100 steps × 2 drift evaluations = **200 UNet forward
passes per batch** (forward-only, no autograd). The inference sbatch allows 4 h.

### 6.6 Standalone minimal script

Complete inference on one field, without the repo's batch scaffolding:

```python
import argparse, yaml, numpy as np, torch
from runners.stochastic_interpolant import StochasticInterpolant
from runners.rs256_guided_diffusion import StdScaler

def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

cfg = dict2namespace(yaml.safe_load(open("configs/kmflow_re1000_rs256_conditional.yml")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.device = device

si = StochasticInterpolant(cfg, device)
ck = torch.load("pretrained_weights/si_ckpt.pth", map_location=device)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else (
     ck[-1] if isinstance(ck, (list, tuple)) else ck)
si.model.load_state_dict(sd)
si.model.eval()

scaler = StdScaler(0.0, 4.786953449708109)          # §6.3 — no 3.4 GB file needed

u  = np.load("data/kmflow_sampled_data_irregnew.npz")["u3232"]
x0 = scaler(torch.as_tensor(u[36, 10:13][None].astype(np.float32)).to(device))  # (1,3,256,256)

with torch.no_grad():
    X = si.sample(x0, n_steps=100)
out = scaler.inverse(X)          # (1,3,256,256) physical vorticity
```

**Input requirements:** `(B, 3, 256, 256)`, float32, where the 3 channels are consecutive
frames. 256×256 is **hard-asserted** — any other resolution raises an `AssertionError`.

---

## 7. Verifying it worked

Measured over 5 of the 64 test batches (100 frames), reference std = 4.963:

| method | mean | std | RMSE vs ref | corr vs ref |
|---|---|---|---|---|
| reference | −0.000 | **4.963** | — | — |
| input (x₀) | 0.028 | 4.954 | 2.059 | 0.914 |
| baseline (physics-guided diffusion) | 0.025 | 4.236 | 1.548 | 0.956 |
| DPS | 0.028 | 4.698 | 1.475 | 0.955 |
| **SI** | **0.000** | **4.945** | **0.724** | **0.989** |

**Expected behaviour of a correct SI run:**

- RMSE vs reference ≈ **0.7** — roughly **half** the baseline's, and well below the input's 2.06.
- **std ≈ 4.94**, i.e. within ~0.4 % of the reference. This is the key signal: SI preserves
  fine-scale energy, whereas the baseline is over-smoothed (std 4.24, a ~15 % variance deficit).
- The sample image should show structured turbulence with fine detail, not noise and not a
  blur.

An **under-trained** model produces a recognisable but **blurry, wavy** field — that is what
100 epochs looks like, and it is normal.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError: SI checkpoint not found` | train first, or point `--si_ckpt` at the right path |
| `load_state_dict` size mismatch | the `--config` differs from the one used in training; architecture is rebuilt from config |
| `AssertionError` in `ConditionalModel.forward` | input is not 256×256 |
| `CUDA out of memory` (training) | lower `--batch_size` (32 → 16 → 8). `--mem` is **CPU** RAM and won't help |
| Run "did nothing", folders unchanged | **`main.py` swallows exceptions** (`try/except → logging.error(traceback)`) and still exits 0. `grep -i traceback` the log |
| Folder still has `metric_log_*.pkl` | the SI run never wrote there — SI wipes each `sample_batch` dir. You are looking at old `reconstruct()` output |
| Results identical to the baseline | `--run_si 1` was not passed. Check the log's first line for the routing message |
| `loss=nan` | diverged; lower `--lr` |
| Animation trembles frame to frame | expected — see §9 |

---

## 9. Known limitations

**This model is a Re=1000, 256×256, 1024-sparse-sensor Kolmogorov-flow specialist.**
There is no Reynolds-number input; Re is baked into the training distribution. On data at a
different Re it would confidently produce **Re=1000-looking structures** — plausible and
wrong. It also will not transfer to a different sensor count, a different degradation type,
a different resolution (hard-asserted), or a different flow. Generalising would require
conditioning on Re and training across degradations.

**Temporal flicker.** An animation of consecutive frames trembles. This is expected, not a
bug: each 3-frame stack is generated **independently** with its own noise, so consecutive
frames are different valid draws from `p(x₁|x₀)`. There is no temporal coupling between
stacks. SI shows this *more* than the baseline precisely because it is less smoothed — the
fine scales it recovers are the genuinely uncertain part given 1.56 % coverage. The paper
evaluates on **temporally decorrelated snapshots** and never claims temporal coherence.

---

## 10. Deviations from the paper

| Schiødt et al. (2026) | This implementation |
|---|---|
| velocity `(u,v)`, 2 channels, 128×128 | **vorticity**, 3 time-frames as channels, 256×256 |
| spectral lowpass (k=8) → 16×16 → cubic upsample | **1024 irregular sensors**, nearest-neighbour filled |
| ConvNeXt-based UNet, GELU | this repo's ResNet + attention UNet (`ConditionalModel`) |
| Eq 9: fixed grid of `N_τ` pseudo-times | one **random τ per sample per step** (equivalent estimator) |
| 4000 epochs, batch 40, 2000 pairs | 2000 epochs, batch 32, 2880 pairs |
| **divergence-free Helmholtz–Hodge projection** | **omitted** — it is defined for velocity fields; this data is vorticity |
| **patch-wise `SI_patch`** (free + cond generators) | **full-field only** (the `SI_full` analogue) |

The last two are the paper's own physics-fidelity and scalability mechanisms. If SI's
**physical** metrics need improvement, those are the first things to add — not more epochs.

---

## 11. Provenance

- Implemented on branch `feature/stochastic-interpolants`.
- Paper: `Schiødt, Mücke & Velte, Sci. Rep. 16:4229 (2026)`, doi:10.1038/s41598-025-34363-y
- Interpolant math verified numerically: `I₀ = x₀` and `I₁ = x₁` exactly; the Heun
  integrator converges to `x₁` under the ideal drift.
- Trained once, 2000 epochs, seed 1234, producing the 40 MB `si_ckpt.pth` documented above.
