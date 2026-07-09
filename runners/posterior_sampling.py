import os
import sys


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
        """Forward operator A(x): high-res field -> low-res measurement."""
        return F.interpolate(
            x, scale_factor=1.0 / scale_factor, mode="bicubic", align_corners=False
        )

    def _build_timesteps(self):
        total_noise_levels = min(self.args.t, self.num_timesteps)
        num_reverse_steps = min(self.args.reverse_steps, total_noise_levels)
        skip = max(1, total_noise_levels // num_reverse_steps)
        return list(range(0, total_noise_levels, skip))

    def dps_sample(self, low_res_measurement, zeta=1.0):
        """Reconstruct high-res flow from noise guided by low-res measurement y."""
        y = low_res_measurement.to(self.device)
        batch_size = y.shape[0]
        high_res_shape = (
            batch_size,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
        )
        x = torch.randn(high_res_shape, device=self.device)
        timesteps = self._build_timesteps()

        self.log(
            f"Starting Stable DPS reverse sampling: {len(timesteps)} steps, "
            f"scale_factor={self.args.scale_factor}, zeta={zeta}"
        )

        for i in tqdm(reversed(timesteps), total=len(timesteps), desc="DPS sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = x.detach().requires_grad_(True)

            alpha_bar_t = self.alphas_bar[i]
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]

            if self.config.model.type == "conditional":
                noise_pred = self.model(x, t, dx=None)
            else:
                noise_pred = self.model(x, t)

            x0_hat = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

            y_hat = self.fluid_downsample_operator(x0_hat, scale_factor=self.args.scale_factor)
            

            loss = torch.norm(y - y_hat, p=2)
            

            guidance_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
            
            # Nan guard safety check
            if torch.isnan(guidance_grad).any():
                guidance_grad = torch.nan_to_num(guidance_grad, nan=0.0)

            with torch.no_grad():
            
                x_mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
                )

   
                x_mean = x_mean - zeta * guidance_grad

                if i > 0:
                    prev_idx = max(i - 1, 0)
                    variance = beta_t * (1.0 - self.alphas_bar[prev_idx]) / (1.0 - alpha_bar_t)
                    x = x_mean + torch.sqrt(variance) * torch.randn_like(x)
                else:
                    x = x_mean

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
        testset = torch.utils.data.TensorDataset(blur_data, ref_data)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run))

        for batch_index, (blur_batch, gt_batch) in enumerate(test_loader):
            # batch_index starts at 0, so folder will be sample_batch1, sample_batch2...
            self.log(f"Batch {batch_index + 1}/{len(test_loader)}")

            gt = gt_batch.to(self.device)
            blur = blur_batch.to(self.device)
            blur_scaled = scaler(blur)
            y = self.fluid_downsample_operator(
                blur_scaled, scale_factor=self.args.scale_factor
            )

            # Clean chronological naming for folders
            sample_folder = f"sample_batch{batch_index + 1}"
            batch_dir = os.path.join(self.log_dir, sample_folder)
            ensure_dir(batch_dir)

            make_image_grid(slice2sequence(blur), os.path.join(batch_dir, "input_image.png"))
            make_image_grid(slice2sequence(gt), os.path.join(batch_dir, "reference_image.png"))

            l2_init = l2_loss(blur, gt)
            self.log(f"L2 loss init: {l2_init}")

            for repeat in range(self.args.repeat_run):
                self.log(f"Run {repeat + 1}/{self.args.repeat_run}")
                sample = self.dps_sample(y, zeta=self.args.zeta)
                sample = scaler.inverse(sample)

                l2_final = l2_loss(sample, gt)
                residual_final = voriticity_residual(sample, calc_grad=False).detach()
                self.log(f"L2 loss final: {l2_final}")
                self.log(f"Residual final: {residual_final}")

                start = batch_index * blur_batch.shape[0]
                end = start + blur_batch.shape[0]
                l2_loss_all[start:end, repeat] = l2_final.item()

        
                if self.config.sampling.dump_arr:
                    np.save(
                        os.path.join(batch_dir, f"sample_arr_run_{batch_index + 1}.npy"),
                        slice2sequence(sample).cpu().numpy(),
                    )

                make_image_grid(
                    slice2sequence(sample),
                    os.path.join(batch_dir, f"sample_run_{batch_index + 1}.png"),
                )

        self.log("Finished DPS sampling")
        self.log(f"Mean L2 loss: {l2_loss_all[..., -1].mean()}")