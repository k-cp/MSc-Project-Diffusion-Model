"""Stochastic Interpolants (SI) for turbulent-flow super-resolution.

Implements the method of Schiodt, Mucke & Velte (Sci. Reports 2026,
"Generative super-resolution of turbulent flows via stochastic interpolants"),
adapted to this repo's data:

  * base sample   x0 = standardized low-res field (the coarse observation)
  * target sample x1 = standardized high-res ground truth

The drift network b_theta(I_tau, x0, tau) is parameterized by reusing this
repo's ConditionalModel UNet (state -> x, pseudo-time -> t, conditional -> dx).

References to paper equations:
  * interpolant           I_tau = alpha*x0 + beta*x1 + sigma*W_tau      (Eq 5)
  * boundary conditions   alpha0=beta1=1, alpha1=beta0=sigma1=0         (Eq 6)
  * generative SDE        dX = b(X,x0) dtau + sigma dW, X_0 = x0        (Eq 7)
  * drift-matching loss   ||b(I_tau,x0) - R_tau||^2, R = a'x0+b'x1+s'W  (Eq 8-9)
  * coefficients          alpha=1-tau, beta=tau^2, sigma=0.1(1-tau)     (Eq 13)

Note: the paper's divergence-free (Helmholtz-Hodge) projection is defined for
velocity fields and is intentionally omitted here (this data is vorticity).
"""

import os
import sys
import math


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging

from models.diffusion_new import ConditionalModel, Model
from runners.rs256_guided_diffusion import (
    StdScaler,
    ensure_dir,
    l2_loss,
    load_recons_data,
    make_image_grid,
    slice2sequence,
    voriticity_residual,
)


class InterpolantCoefficients:
    """Interpolant coefficients and their pseudo-time derivatives (Eq 13).

    alpha(tau) = 1 - tau,  beta(tau) = tau^2,  sigma(tau) = 0.1 (1 - tau).
    Satisfies the boundary conditions (Eq 6): I_0 = x0, I_1 = x1.
    """

    sigma_scale = 0.1

    def alpha(self, tau):
        return 1.0 - tau

    def alpha_dot(self, tau):
        return -torch.ones_like(tau)

    def beta(self, tau):
        return tau ** 2

    def beta_dot(self, tau):
        return 2.0 * tau

    def sigma(self, tau):
        return self.sigma_scale * (1.0 - tau)

    def sigma_dot(self, tau):
        return -self.sigma_scale * torch.ones_like(tau)


def load_si_pairs(ref_path, sample_path, data_kw, split="train", frame_stride=1):
    """Load paired (x0=low-res, x1=high-res) 3-frame stacks.

    Selects the
    train split (all but the last 4 trajectories) or the test split.
    Standardization stats are always computed from the train portion so train
    and test use the same scaler. frame_stride>1 subsamples frames within each
    trajectory to reduce dataset size and decorrelate samples (paper uses
    temporally decorrelated snapshots).
    """
    with np.load(sample_path, allow_pickle=True) as f:
        sampled_all = np.asarray(f[data_kw]).astype(np.float32)
    ref_all = np.load(ref_path).astype(np.float32)

    data_mean, data_std = np.mean(ref_all[:-4]), np.std(ref_all[:-4])

    if split == "train":
        ref_sel, samp_sel = ref_all[:-4], sampled_all[:-4]
    elif split == "test":
        ref_sel, samp_sel = ref_all[-4:], sampled_all[-4:]
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    ref_sel = torch.as_tensor(ref_sel.copy(), dtype=torch.float32)
    samp_sel = torch.as_tensor(samp_sel.copy(), dtype=torch.float32)

    x1_list, x0_list = [], []

    # Grab 3 consecutive frames
    for i in range(ref_sel.shape[0]):
        for j in range(0, ref_sel.shape[1] - 2, frame_stride):
            x1_list.append(ref_sel[i, j:j + 3])   # high-res target
            x0_list.append(samp_sel[i, j:j + 3])  # low-res base
    x1 = torch.stack(x1_list, dim=0)
    x0 = torch.stack(x0_list, dim=0)

    # x0: The batched tensor of low-res (blurred) inputs. [Batch Size, 3, 256, 256]

    # x1: The batched tensor of high-res (target) outputs.[ Batch Size, 3, 256, 256]


    return x0, x1, float(data_mean), float(data_std)


def load_si_targets(ref_path, split="train", frame_stride=1):
    """Load high-res target stacks x1 ONLY (for blind training that manufactures
    x0 on the fly). Avoids loading the 6.7 GB sensor file. Returns
    (x1_stacks, mean, std) with stats from the train split, as in load_si_pairs.
    """
    ref_all = np.load(ref_path).astype(np.float32)
    data_mean, data_std = np.mean(ref_all[:-4]), np.std(ref_all[:-4])
    ref_sel = ref_all[:-4] if split == "train" else ref_all[-4:]
    ref_sel = torch.as_tensor(ref_sel.copy(), dtype=torch.float32)

    stacks = []
    for i in range(ref_sel.shape[0]):
        for j in range(0, ref_sel.shape[1] - 2, frame_stride):
            stacks.append(ref_sel[i, j:j + 3])
    return torch.stack(stacks, dim=0), float(data_mean), float(data_std)


class DegradationSampler:
    """Manufacture a coarse observation x0 from a high-res field x1 with a
    RANDOMLY SAMPLED degradation ('blind' SI training, Option 1).

    The point: feed the drift network a different degradation every step so it
    learns a *family* of inverse maps instead of overfitting one fixed operator
    (the frozen u3232). Both the degradation *family* and its *parameters*
    (crucially, the sensor count) are re-rolled per sample.

    Families:
      * 'sensor'     -- N random point sensors (N ~ U[n_min, n_max]),
                        nearest-neighbour (Voronoi) filled -- matches how u3232
                        is built. N is random so the model is robust to how much
                        low-res data it gets, not just the layout.
      * 'downsample' -- uniform factor from down_factors, nearest down+up.
      * 'lowpass'    -- spectral lowpass at a cutoff from lowpass_cutoffs.

    Operates on one (C, H, W) numpy stack; the C channels are the 3 frames and
    share the same spatial degradation (as real sensors are fixed across frames).
    Uses np.random.* so DataLoader workers (seeded per worker via
    si_worker_init_fn) stay independent.
    """

    def __init__(self, families=("sensor",), n_min=256, n_max=4000,
                 down_factors=(4, 8, 16), lowpass_cutoffs=(4, 8, 16), meas_noise=0.0):
        self.families = tuple(families)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.down_factors = tuple(int(f) for f in down_factors)
        self.lowpass_cutoffs = tuple(float(k) for k in lowpass_cutoffs)
        self.meas_noise = float(meas_noise)

    def __call__(self, x1):
        fam = self.families[np.random.randint(len(self.families))]
        if fam == "sensor":
            x0 = self._sensor(x1)
        elif fam == "downsample":
            x0 = self._downsample(x1)
        elif fam == "lowpass":
            x0 = self._lowpass(x1)
        else:
            raise ValueError(f"unknown degradation family {fam!r}")
        if self.meas_noise > 0.0:
            x0 = x0 + np.random.normal(0.0, self.meas_noise, x0.shape).astype(np.float32)
        return x0.astype(np.float32)

    def _sensor(self, x1):
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as e:
            raise ImportError("blind SI 'sensor' degradation needs scipy "
                              "(pip install scipy)") from e
        c, h, w = x1.shape
        n = min(np.random.randint(self.n_min, self.n_max + 1), h * w)
        flat = np.random.choice(h * w, size=n, replace=False)
        mask = np.zeros(h * w, dtype=bool)
        mask[flat] = True
        mask = mask.reshape(h, w)
        # nearest measured pixel for every location (exact Euclidean, like u3232)
        iy, ix = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        return x1[:, iy, ix]

    def _downsample(self, x1):
        c, h, w = x1.shape
        f = self.down_factors[np.random.randint(len(self.down_factors))]
        small = x1[:, ::f, ::f]                                  # nearest subsample
        up = np.repeat(np.repeat(small, f, axis=1), f, axis=2)   # nearest upsample
        return up[:, :h, :w]

    def _lowpass(self, x1):
        c, h, w = x1.shape
        kc = self.lowpass_cutoffs[np.random.randint(len(self.lowpass_cutoffs))]
        kx = np.fft.fftfreq(h) * h
        ky = np.fft.fftfreq(w) * w
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        keep = ((np.abs(KX) <= kc) & (np.abs(KY) <= kc))[None]
        xh = np.fft.fft2(x1, axes=(-2, -1)) * keep
        return np.fft.ifft2(xh, axes=(-2, -1)).real


class SIPairsDataset(torch.utils.data.Dataset):
    """Blind-SI training set: holds high-res stacks, manufactures a fresh
    degraded x0 on every __getitem__ (so num_workers overlap it with the GPU).
    Returns RAW (x0, x1); the training loop standardizes both."""

    def __init__(self, x1_stacks, degrader):
        self.x1 = x1_stacks
        self.degrader = degrader

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, i):
        x1 = self.x1[i]
        x0 = torch.from_numpy(self.degrader(x1.numpy()).copy())
        return x0, x1


def si_worker_init_fn(worker_id):
    """Give each DataLoader worker an independent numpy RNG stream, re-seeded
    each epoch (torch sets a distinct base seed per worker per epoch)."""
    np.random.seed(torch.initial_seed() % (2 ** 32))


class StochasticInterpolant:
    """Drift network + interpolant math + SDE sampler."""

    def __init__(self, config, device, logger=None, physics="none"):
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger("LOG")
        self.coeff = InterpolantCoefficients()
        self.physics = physics

        # Reuse the conditional UNet as the drift network b_theta.
        if config.model.type == "conditional":
            self.model = ConditionalModel(config)
        else:
            self.model = Model(config)

        # "learned" physics conditioning (Shu et al. method 1) feeds BOTH the
        # coarse field x0 and the PDE residual gradient c through the
        # conditioning branch, so emb_conv must ingest 2*in_channels. Patched
        # here rather than in models/diffusion_new.py so the diffusion model
        # used by baseline/DPS is untouched (its checkpoint still loads).
        if self.physics == "learned":
            ch = config.model.ch
            in_ch = config.model.in_channels
            self.model.emb_conv = nn.Sequential(
                nn.Conv2d(2 * in_ch, ch, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1,
                          padding_mode="circular"),
            )

        self.model.to(device)

        # Pseudo-time tau in [0,1] is scaled into the timestep-embedding range
        # (same scale must be used in training and sampling).
        self.time_scale = float(config.diffusion.num_diffusion_timesteps)

    def log(self, msg):
        self.logger.info(msg)

    def physics_grad(self, x_std, scaler):
        """PDE residual gradient c = dr/dx in standardized units.

        Mirrors reconstruct()'s physical_gradient_func exactly, so 'linear' and
        'learned' here use the same quantity the baseline uses. The residual is
        evaluated on the CURRENT state (Shu et al. Eq 6: r_t = G|u=x_t), which
        suits SI even better than diffusion -- the interpolant carries only ~4%
        noise at its peak, so it is far closer to a physical field than a
        diffusion x_t.

        voriticity_residual() calls torch.autograd.grad internally, so grad must
        be enabled even when the caller is under torch.no_grad().
        """
        with torch.enable_grad():
            return voriticity_residual(scaler.inverse(x_std))[0] / scaler.scale()

    def drift(self, x_state, tau_vec, x0_cond, phys_cond=None):
        """b_theta(x_state, tau, x0_cond[, c]). tau_vec: (B,) pseudo-time in [0,1].

        physics='learned': the conditioning branch receives cat([x0, c]); passing
        phys_cond=None gives the unconditional branch (c = empty set), which is what
        classifier-free guidance needs.
        """
        t = tau_vec.float() * self.time_scale
        if self.config.model.type != "conditional":
            return self.model(x_state, t)

        cond = x0_cond
        if self.physics == "learned":
            if phys_cond is None:
                phys_cond = torch.zeros_like(x0_cond)
            cond = torch.cat([x0_cond, phys_cond], dim=1)
        return self.model(x_state, t, dx=cond)

    def interpolant_loss(self, x0, x1, scaler=None, p_uncond=0.1):
        """Drift-matching objective (Eq 9), estimated on a minibatch.

        physics='learned' additionally computes c = dr/dI_tau and feeds it to the
        conditioning branch, dropping it with probability p_uncond so the
        unconditional branch is trained too (Shu et al. Algorithm 1, line 6).
        """
        b = x0.shape[0]
        tau = torch.rand(b, 1, 1, 1, device=x0.device)          # tau ~ U[0,1]
        z = torch.randn_like(x0)
        w = torch.sqrt(tau) * z                                  # Wiener W_tau ~ N(0, tau)

        c = self.coeff
        interpolant = c.alpha(tau) * x0 + c.beta(tau) * x1 + c.sigma(tau) * w
        target = c.alpha_dot(tau) * x0 + c.beta_dot(tau) * x1 + c.sigma_dot(tau) * w

        phys_cond = None
        if self.physics == "learned":
            if scaler is None:
                raise ValueError("physics='learned' training requires a scaler")
            phys_cond = self.physics_grad(interpolant.detach(), scaler)
            if torch.rand(1).item() < p_uncond:      # unconditional dropout
                phys_cond = None

        pred = self.drift(interpolant, tau.reshape(b), x0, phys_cond)
        return ((pred - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, x0, n_steps=100, scaler=None, lam=0.0, w_cond=0.0):
        """Integrate the generative SDE (Eq 7) from X_0 = x0 with a stochastic
        Heun (predictor-corrector) scheme, returning X_1.

        Physics guidance (self.physics), following Shu et al. (JCP 2023):
          'none'    -- plain SI.
          'linear'  -- direct gradient descent of the physics-informed condition:
                       subtract lam * c from the update (their Eq 9, the "Linear"
                       variant). Inference-only; works with a plain SI checkpoint.
          'learned' -- learned encoding of the physics-informed condition: c is fed
                       to the conditioning branch and combined classifier-free with
                       strength w_cond (their Algorithm 2, line 8). REQUIRES a
                       checkpoint trained with physics='learned'.
        """
        b = x0.shape[0]
        x = x0.clone()
        taus = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)

        if self.physics != "none" and scaler is None:
            raise ValueError(f"physics='{self.physics}' sampling requires a scaler")

        def _drift(state, tau_scalar):
            """One drift evaluation, applying 'learned' guidance if enabled."""
            tvec = torch.full((b,), tau_scalar, device=x0.device)
            if self.physics != "learned":
                return self.drift(state, tvec, x0)
            c = self.physics_grad(state, scaler)
            b_cond = self.drift(state, tvec, x0, c)
            if w_cond == 0.0:
                return b_cond
            b_uncond = self.drift(state, tvec, x0, None)
            return b_cond + w_cond * (b_cond - b_uncond)

        for i in range(n_steps):
            tau = taus[i].item()
            tau_next = taus[i + 1].item()
            dtau = tau_next - tau
            sig = self.coeff.sigma_scale * (1.0 - tau)
            noise = sig * math.sqrt(dtau) * torch.randn_like(x)

            drift1 = _drift(x, tau)
            x_pred = x + drift1 * dtau + noise

            drift2 = _drift(x_pred, tau_next)
            x = x + 0.5 * (drift1 + drift2) * dtau + noise

            # 'linear': direct gradient descent on the PDE residual (Eq 9).
            if self.physics == "linear" and lam != 0.0:
                x = x - lam * self.physics_grad(x, scaler)

        return x


class SIRunner:
    """Inference: super-resolve the test set with a trained SI drift network.

    Mirrors the DPS/baseline output layout so animate_results.py / metrics.py
    can consume it (sample_batch<i>/ with input, reference, sample as png + npy).
    """

    def __init__(self, args, config, logger=None, log_dir=None):
        self.args = args
        self.config = config
        self.logger = logger or logging.getLogger("LOG")
        self.log_dir = log_dir or config.log_dir
        self.device = config.device
        self.n_steps = getattr(args, "si_steps", 100)
        self.physics = getattr(args, "si_physics", "none")
        self.lam = getattr(args, "si_lambda", 0.0)
        self.w_cond = getattr(args, "si_w", 0.0)

        self.si = StochasticInterpolant(config, self.device, self.logger,
                                        physics=self.physics)

        ckpt_path = getattr(args, "si_ckpt", None) or "./pretrained_weights/si_ckpt.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"SI checkpoint not found: {ckpt_path}. Train one first with train_si.py."
            )
        states = torch.load(ckpt_path, map_location=self.device)
        if isinstance(states, dict) and "model" in states:
            state_dict = states["model"]            # {'epoch','model','optimizer'} (train_si.py)
        elif isinstance(states, (list, tuple)):
            state_dict = states[-1]                  # [state_dict] legacy format
        else:
            state_dict = states                      # raw state_dict
        self.si.model.load_state_dict(state_dict)
        self.si.model.eval()
        self.log(f"Loaded SI drift network from {ckpt_path}")
        if self.physics == "linear":
            self.log(f"Physics guidance: linear (direct gradient descent), lambda={self.lam}")
        elif self.physics == "learned":
            self.log(f"Physics guidance: learned (conditioned), w={self.w_cond}")
        else:
            self.log("Physics guidance: none")

    def log(self, msg):
        self.logger.info(msg)

    def si_sample_pipeline(self):
        import shutil

        self.log("Loading reconstruction data for Stochastic Interpolants")
        ref_data, blur_data, data_mean, data_std = load_recons_data(
            self.config.data.data_dir,
            self.config.data.sample_data_dir,
            self.config.data.data_kw,
            smoothing=self.config.data.smoothing,
            smoothing_scale=self.config.data.smoothing_scale,
        )

        scaler = StdScaler(data_mean, data_std)
        self.log(f"SI outputs -> {self.log_dir}  (sample_batch<i> folders)")

        testset = torch.utils.data.TensorDataset(blur_data, ref_data)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run))       
    


        for batch_index, (blur_batch, gt_batch) in enumerate(test_loader):
            self.log(f"Batch {batch_index + 1}/{len(test_loader)}")

            gt = gt_batch.to(self.device)
            blur = blur_batch.to(self.device)
            x0 = scaler(blur)  # base sample = standardized low-res field

            batch_dir = os.path.join(self.log_dir, f"sample_batch{batch_index}")
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
            ensure_dir(batch_dir)

            make_image_grid(slice2sequence(blur), os.path.join(batch_dir, "input_image.png"))
            make_image_grid(slice2sequence(gt), os.path.join(batch_dir, "reference_image.png"))
            if self.config.sampling.dump_arr:
                np.save(os.path.join(batch_dir, "input_arr.npy"),
                        slice2sequence(blur).cpu().numpy())
                np.save(os.path.join(batch_dir, "reference_arr.npy"),
                        slice2sequence(gt).cpu().numpy())

            l2_init = l2_loss(blur, gt)
            self.log(f"L2 loss init: {l2_init}")

            for repeat in range(self.args.repeat_run):
                self.log(f"Run {repeat + 1}/{self.args.repeat_run} (SDE steps={self.n_steps})")
                sample = self.si.sample(
                    x0,
                    n_steps=self.n_steps,
                    scaler=scaler,
                    lam=self.lam,
                    w_cond=self.w_cond,
                )
                sample = scaler.inverse(sample)

                l2_final = l2_loss(sample, gt)
                residual_final = voriticity_residual(sample, calc_grad=False).detach()
                self.log(f"L2 loss final: {l2_final}")
                self.log(f"Residual final: {residual_final}")

                start = batch_index * blur_batch.shape[0]
                end = start + blur_batch.shape[0]
                l2_loss_all[start:end, repeat] = l2_final.item()

                make_image_grid(
                    slice2sequence(sample),
                    os.path.join(batch_dir, f"sample_run_{repeat}_it0.png"),
                )
                if self.config.sampling.dump_arr:
                    np.save(
                        os.path.join(batch_dir, f"sample_arr_run_{repeat}_it0.npy"),
                        slice2sequence(sample).cpu().numpy(),
                    )

        self.log("Finished SI sampling")
        self.log(f"Mean L2 loss: {l2_loss_all[..., -1].mean()}")
