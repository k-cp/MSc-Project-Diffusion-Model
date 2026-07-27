import os
import sys
import shutil


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from models.diffusion_new import ConditionalModel, Model
from runners.rs256_guided_diffusion import (
    add_measurement_noise,
    StdScaler,
    ensure_dir,
    get_beta_schedule,
    l2_loss,
    load_recons_data,
    make_image_grid,
    slice2sequence,
    voriticity_residual,
)


class PosteriorRunner:
    def __init__(self, args, config, logger=None, log_dir=None):
        self.args = args
        self.config = config
        self.logger = logger or logging.getLogger("LOG")
        self.log_dir = log_dir or config.log_dir
        self.device = config.device

        if self.config.model.type == "conditional":
            self.model = ConditionalModel(self.config)
        else:
            self.model = Model(self.config)

        states = torch.load(self.config.model.ckpt_path, map_location=self.device)
        state_dict = states[-1] if isinstance(states, (list, tuple)) else states
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        betas = get_beta_schedule(
            beta_start=self.config.diffusion.beta_start,
            beta_end=self.config.diffusion.beta_end,
            num_diffusion_timesteps=self.config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def log(self, message):
        self.logger.info(message)

    def fluid_downsample_operator(self, x, scale_factor=4):
        """Bicubic downsample: high-res field -> low-res grid measurement."""
        return F.interpolate(
            x, scale_factor=1.0 / scale_factor, mode="bicubic", align_corners=False
        )

    def sensor_operator(self, x, sensor_idx):
        """Sparse-sensor forward operator A(x): sample x at the given sensor
        locations. sensor_idx: (B, n_sensors) flat indices into the H*W grid.
        Returns (B, C, n_sensors) -- the true measurement for this dataset.
        """
        b, c = x.shape[0], x.shape[1]
        flat = x.reshape(b, c, -1)                     # (B, C, H*W)
        gather_idx = sensor_idx.unsqueeze(1).expand(-1, c, -1)  # (B, C, n_sensors)
        return torch.gather(flat, 2, gather_idx)

    def apply_forward(self, x, sensor_idx=None):
        """Dispatch to the configured forward operator A. Using ONE method for
        both the measurement y = A(gt) and the guidance term A(x0_hat)
        guarantees they are consistent (a hard requirement for DPS)."""
        if self.args.operator == "sparse":
            return self.sensor_operator(x, sensor_idx)
        return self.fluid_downsample_operator(x, scale_factor=self.args.scale_factor)

    def _alpha_bar(self, idx):
        """Cumulative product alpha-bar at a scheduled index; idx == -1 -> 1.0.

        Mirrors functions.denoising_step.compute_alpha so the strided reverse
        chain uses the same schedule bookkeeping as the repo's DDIM/DDPM samplers.
        """
        if idx < 0:
            return torch.tensor(1.0, device=self.device)
        return self.alphas_bar[idx]

    def _build_timesteps(self):
        total_noise_levels = min(self.args.t, self.num_timesteps)
        num_reverse_steps = min(self.args.reverse_steps, total_noise_levels)
        skip = max(1, total_noise_levels // num_reverse_steps)
        return list(range(0, total_noise_levels, skip))

    def dps_sample(self, low_res_measurement, init_field, sensor_idx=None, zeta=1.0,
                   scaler=None, phys_mode="none", phys_lambda=0.0, use_cond=False):
        """Reconstruct a high-res flow field guided by the low-res measurement y.

        Follows Diffusion Posterior Sampling (Chung et al., 2023, Alg. 1) using
        the repo's own strided reverse-diffusion bookkeeping:
          - x0_hat is predicted from the network's noise estimate (no [-1, 1]
            clamp: this data is standardized vorticity, not [-1, 1] pixels);
          - the prior step is the ancestral DDPM update evaluated between two
            *scheduled* timesteps (so strided schedules stay self-consistent);
          - measurement guidance uses the normalized DPS step
            zeta_i = zeta / ||y - A(x0_hat)|| on the gradient of the L2 *norm*.
        """
        y = low_res_measurement.to(self.device)
        batch_size = y.shape[0]

        # Timestep pairs (i -> j) matching functions.denoising_step: j is the
        # next lower scheduled step, or -1 (=> clean signal) on the last step.
        seq = self._build_timesteps()
        seq_next = [-1] + list(seq[:-1])

        # Warm start: diffuse the observed coarse field to the top noise level,
        # consistent with a partial-t (t < num_timesteps) reverse schedule.
        i_max = seq[-1]
        abar_top = self._alpha_bar(i_max)
        x = torch.sqrt(abar_top) * init_field.to(self.device) + \
            torch.sqrt(1.0 - abar_top) * torch.randn_like(init_field, device=self.device)

        if (phys_mode != "none" or use_cond) and scaler is None:
            raise ValueError("dps_physics/dps_cond require a scaler "
                             "(voriticity_residual needs unscaled fields)")
        first_step = True   # so the gradient-scale line is logged once, not 20x

        self.log(
            f"Starting DPS reverse sampling: {len(seq)} steps, "
            f"scale_factor={self.args.scale_factor}, zeta={zeta}"
            + (", physics-conditioned network" if use_cond else "")
            + (f", physics force={phys_mode} (lambda={phys_lambda})" if phys_mode != "none" else "")
        )

        for i, j in tqdm(list(zip(reversed(seq), reversed(seq_next))),
                         total=len(seq), desc="DPS sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = x.detach().requires_grad_(True)

            at = self._alpha_bar(i)       # alpha_bar_t
            atm1 = self._alpha_bar(j)     # alpha_bar_{t_next}
            beta_t = 1.0 - at / atm1      # effective beta over the (i -> j) jump

            if self.config.model.type == "conditional":
                # JCP's LEARNED ENCODING: the checkpoint was trained with the PDE
                # residual gradient as a conditioning input, but plain DPS discards
                # it (dx=None). Switching it on is free -- no extra force, no
                # strength to tune -- it just changes the noise prediction.
                # Evaluated on the noisy x, exactly as the baseline does, so the
                # only remaining difference from the baseline is the guidance term.
                dx_cond = None
                if use_cond:
                    dx_cond = voriticity_residual(
                        scaler.inverse(x.detach()))[0] / scaler.scale()
                noise_pred = self.model(x, t, dx=dx_cond)
            else:
                noise_pred = self.model(x, t)

            # Tweedie estimate of the clean field (no data-range clamp).
            x0_hat = (x - torch.sqrt(1.0 - at) * noise_pred) / torch.sqrt(at)

            y_hat = self.apply_forward(x0_hat, sensor_idx=sensor_idx)

            # DPS guidance: per-sample gradient of the measurement L2 norm.
            # Summing per-sample norms keeps each sample's gradient independent,
            # so grad[b] = d||y_b - A(x0_hat)_b|| / dx (no cross-sample coupling).
            per_sample_norm = torch.linalg.norm((y - y_hat).reshape(batch_size, -1), dim=1)
            # 'x0hat' physics needs a SECOND backward over the same graph, so the
            # first pass must not free it.
            guidance_grad = torch.autograd.grad(outputs=per_sample_norm.sum(), inputs=x,
                                                retain_graph=(phys_mode == "x0hat"))[0]

            # NaN guard.
            if torch.isnan(guidance_grad).any():
                guidance_grad = torch.nan_to_num(guidance_grad, nan=0.0)

            # ---- optional Navier-Stokes term (the DPS+physics hybrid) ----------
            # Plain DPS pins the 1024 measured points and leaves the other 98.4%
            # of the field unconstrained, which is why its residual explodes.
            # This adds the physics the baseline has and DPS lacks.
            phys_grad = None
            if phys_mode == "xt":
                # JCP-style: residual of the NOISY field, differentiated w.r.t.
                # itself inside voriticity_residual. Cheap -- no network graph.
                phys_grad = voriticity_residual(
                    scaler.inverse(x.detach()))[0] / scaler.scale()
            elif phys_mode == "x0hat":
                # DPS-style: evaluate the residual on the CLEAN estimate, then
                # chain back to x through the network -- the same reasoning that
                # makes DPS use p(y|x0_hat) instead of p(y|x_t).
                res_loss = voriticity_residual(scaler.inverse(x0_hat), calc_grad=False)
                phys_grad = torch.autograd.grad(outputs=res_loss, inputs=x)[0]
            if phys_grad is not None and torch.isnan(phys_grad).any():
                phys_grad = torch.nan_to_num(phys_grad, nan=0.0)

            if first_step and phys_grad is not None:
                # Their scales differ a lot; print both so phys_lambda is tunable.
                self.log(f"  |measurement grad|={guidance_grad.norm().item():.4g}  "
                         f"|physics grad|={phys_grad.norm().item():.4g}  "
                         f"(mode={phys_mode}, lambda={phys_lambda})")
                first_step = False

            with torch.no_grad():
                # Ancestral DDPM prior step between the two scheduled timesteps.
                mean = (
                    torch.sqrt(atm1) * beta_t * x0_hat
                    + torch.sqrt(1.0 - beta_t) * (1.0 - atm1) * x
                ) / (1.0 - at)

                if j >= 0:
                    noise = torch.sqrt(beta_t) * torch.randn_like(x)
                else:
                    noise = 0.0  # no noise injected on the final step
                x_prior = mean + noise

                # Normalized DPS update (Alg. 1): zeta_i = zeta / ||residual||,
                # applied per sample so the step is independent of batch size.
                step = (zeta / (per_sample_norm.detach() + 1e-8)).view(batch_size, 1, 1, 1)
                x = x_prior - step * guidance_grad
                if phys_grad is not None:
                    x = x - phys_lambda * phys_grad
                x = x.detach()

        return x

    def dps_sample_pipeline(self):
        self.log("Loading reconstruction data for DPS")
        ref_data, blur_data, data_mean, data_std = load_recons_data(
            self.config.data.data_dir,
            self.config.data.sample_data_dir,
            self.config.data.data_kw,
            smoothing=self.config.data.smoothing,
            smoothing_scale=self.config.data.smoothing_scale,
        )

        scaler = StdScaler(data_mean, data_std)
        self.log(f"DPS outputs -> {self.log_dir}  (sample_batch<i> folders)")

        # Per-sample sensor indices for the sparse operator, aligned with the
        # trajectory-major flattening done inside load_recons_data (for each of
        # the last 4 test trajectories, every (frames-2) sub-sample shares that
        # trajectory's 1024 sensor locations).
        if self.args.operator == "sparse":
            with np.load(self.config.data.sample_data_dir, allow_pickle=True) as f:
                idx_test = f["idx_lst"][-4:].astype(np.int64)      # (4, n_sensors)
                n_frames = f[self.config.data.data_kw].shape[1]
            per_traj = n_frames - 2
            sensor_idx_all = np.repeat(idx_test, per_traj, axis=0)  # (4*per_traj, n_sensors)
            sensor_idx_all = torch.from_numpy(sensor_idx_all)
            assert sensor_idx_all.shape[0] == ref_data.shape[0], (
                f"sensor idx count {sensor_idx_all.shape[0]} != samples {ref_data.shape[0]}"
            )
            self.log(f"Sparse operator: {sensor_idx_all.shape[1]} sensors per field")
            testset = torch.utils.data.TensorDataset(blur_data, ref_data, sensor_idx_all)
        else:
            testset = torch.utils.data.TensorDataset(blur_data, ref_data)

        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run))

        for batch_index, batch in enumerate(test_loader):
            if self.args.operator == "sparse":
                blur_batch, gt_batch, idx_batch = batch
                sensor_idx = idx_batch.to(self.device)
            else:
                blur_batch, gt_batch = batch
                sensor_idx = None
            # batch_index starts at 0, so folder will be sample_batch1, sample_batch2...
            self.log(f"Batch {batch_index + 1}/{len(test_loader)}")

            gt = gt_batch.to(self.device)
            blur = blur_batch.to(self.device)
            gt_scaled = scaler(gt)
            blur_scaled = scaler(blur)

            # Measurement y = A(x_true): a genuine observation of the ground
            # truth through the SAME forward operator used in the guidance term.
            # (Guiding toward A(blur) would double-degrade and make gt
            # unreachable.) The coarse observed field seeds the warm start.
            y = self.apply_forward(gt_scaled, sensor_idx=sensor_idx)
            # Simulated sensor noise on the MEASUREMENT itself -- the regime DPS
            # was designed for (Chung et al. target noisy inverse problems).
            # Applied to y (what the guidance term matches) and to the warm-start
            # field, so the whole pipeline sees the same corrupted observation.
            _mn = getattr(self.args, "meas_noise", 0.0)
            y = add_measurement_noise(y, _mn, self.args.seed, batch_index)
            blur_scaled = add_measurement_noise(blur_scaled, _mn, self.args.seed, batch_index)

            # One folder per batch, 0-indexed, directly under log_dir -- matching
            # reconstruct() and what animate_results.py / generate_metrics expect.
            # Wipe it first so no stale files from a previous run survive.
            batch_dir = os.path.join(self.log_dir, f"sample_batch{batch_index}")
            if os.path.exists(batch_dir):
                shutil.rmtree(batch_dir)
            ensure_dir(batch_dir)

            # Input (coarse/sparse observation) and reference (ground truth):
            # image + raw data array, with the exact filenames the repo uses.
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
                self.log(f"Run {repeat + 1}/{self.args.repeat_run}")
                sample = self.dps_sample(
                    y, init_field=blur_scaled, sensor_idx=sensor_idx, zeta=self.args.zeta,
                    scaler=scaler,
                    phys_mode=getattr(self.args, "dps_physics", "none"),
                    phys_lambda=getattr(self.args, "dps_lambda", 0.0),
                    use_cond=(getattr(self.args, "dps_cond", 0) == 1),
                )
                sample = scaler.inverse(sample)

                l2_final = l2_loss(sample, gt)
                residual_final = voriticity_residual(sample, calc_grad=False).detach()
                self.log(f"L2 loss final: {l2_final}")
                self.log(f"Residual final: {residual_final}")

                start = batch_index * blur_batch.shape[0]
                end = start + blur_batch.shape[0]
                l2_loss_all[start:end, repeat] = l2_final.item()

                # Sample reconstruction: IMAGE (new -- reconstruct() never made one)
                # + data array named like reconstruct() so animate_results.py works.
                make_image_grid(
                    slice2sequence(sample),
                    os.path.join(batch_dir, f"sample_run_{repeat}_it0.png"),
                )
                if self.config.sampling.dump_arr:
                    np.save(
                        os.path.join(batch_dir, f"sample_arr_run_{repeat}_it0.npy"),
                        slice2sequence(sample).cpu().numpy(),
                    )

        self.log("Finished DPS sampling")
        self.log(f"Mean L2 loss: {l2_loss_all[..., -1].mean()}")
