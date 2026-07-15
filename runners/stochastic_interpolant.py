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

    Mirrors runners.rs256_guided_diffusion.load_recons_data, but selects the
    train split (all but the last 4 trajectories) or the test split (last 4).
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
    for i in range(ref_sel.shape[0]):
        for j in range(0, ref_sel.shape[1] - 2, frame_stride):
            x1_list.append(ref_sel[i, j:j + 3])   # high-res target
            x0_list.append(samp_sel[i, j:j + 3])  # low-res base
    x1 = torch.stack(x1_list, dim=0)
    x0 = torch.stack(x0_list, dim=0)
    return x0, x1, float(data_mean), float(data_std)


class StochasticInterpolant:
    """Drift network + interpolant math + SDE sampler."""

    def __init__(self, config, device, logger=None):
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger("LOG")
        self.coeff = InterpolantCoefficients()

        # Reuse the conditional UNet as the drift network b_theta.
        if config.model.type == "conditional":
            self.model = ConditionalModel(config)
        else:
            self.model = Model(config)
        self.model.to(device)

        # Pseudo-time tau in [0,1] is scaled into the timestep-embedding range
        # (same scale must be used in training and sampling).
        self.time_scale = float(config.diffusion.num_diffusion_timesteps)

    def log(self, msg):
        self.logger.info(msg)

    def drift(self, x_state, tau_vec, x0_cond):
        """b_theta(x_state, tau, x0_cond). tau_vec: (B,) pseudo-time in [0,1]."""
        t = tau_vec.float() * self.time_scale
        if self.config.model.type == "conditional":
            return self.model(x_state, t, dx=x0_cond)
        return self.model(x_state, t)

    def interpolant_loss(self, x0, x1):
        """Drift-matching objective (Eq 9), estimated on a minibatch."""
        b = x0.shape[0]
        tau = torch.rand(b, 1, 1, 1, device=x0.device)          # tau ~ U[0,1]
        z = torch.randn_like(x0)
        w = torch.sqrt(tau) * z                                  # Wiener W_tau ~ N(0, tau)

        c = self.coeff
        interpolant = c.alpha(tau) * x0 + c.beta(tau) * x1 + c.sigma(tau) * w
        target = c.alpha_dot(tau) * x0 + c.beta_dot(tau) * x1 + c.sigma_dot(tau) * w

        pred = self.drift(interpolant, tau.reshape(b), x0)
        return ((pred - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, x0, n_steps=100):
        """Integrate the generative SDE (Eq 7) from X_0 = x0 with a stochastic
        Heun (predictor-corrector) scheme, returning X_1."""
        b = x0.shape[0]
        x = x0.clone()
        taus = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device)

        for i in range(n_steps):
            tau = taus[i].item()
            tau_next = taus[i + 1].item()
            dtau = tau_next - tau
            sig = self.coeff.sigma_scale * (1.0 - tau)
            noise = sig * math.sqrt(dtau) * torch.randn_like(x)

            tvec = torch.full((b,), tau, device=x0.device)
            drift1 = self.drift(x, tvec, x0)
            x_pred = x + drift1 * dtau + noise

            tvec_next = torch.full((b,), tau_next, device=x0.device)
            drift2 = self.drift(x_pred, tvec_next, x0)
            x = x + 0.5 * (drift1 + drift2) * dtau + noise

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

        self.si = StochasticInterpolant(config, self.device, self.logger)

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
                sample = self.si.sample(x0, n_steps=self.n_steps)
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
