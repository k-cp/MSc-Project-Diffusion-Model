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


try:
    from models.diffusion import Model
except ModuleNotFoundError:
    from models.rs256_guided_diffusion import Model



class PosteriorRunner:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.config)
        if hasattr(self.config.model, "ckpt_path"):
            states = torch.load(self.config.model.ckpt_path, map_location=self.device)
            self.model.load_state_dict(states[0]) # Load weights
        self.model.to(self.device)
        self.model.eval()


        betas = get_beta_schedule(config.diffusion.beta_schedule, config.diffusion.num_timesteps)
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]
        
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

    def fluid_downsample_operator(self, x, scale_factor=4):
        """ 
        The forward operator A(x). 
        Simulates how high-res fluid domain turns into a low-res measurement.
        """
        return F.interpolate(x, scale_factor=1.0/scale_factor, mode='bicubic', align_corners=False)

    def dps_sample(self, low_res_measurement, zeta=1.0):
        """
        The main loop. Reconstructs fluid flow from pure noise guided by low_res_measurement.
        """

        y = low_res_measurement.to(self.device)

        batch_size = y.shape[0]
        high_res_shape = (batch_size, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
        x = torch.randn(high_res_shape, device=self.device)
        
        print("Starting DPS Reverse Sampling Loop...")
        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            

            x = x.detach().requires_grad_(True)
            
       
            alpha_bar_t = self.alphas_bar[i]
            beta_t = self.betas[i]
            alpha_t = self.alphas[i]
            

            noise_pred = self.model(x, t)
            


            x0_hat = (x - torch.sqrt(1.0 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

            y_hat = self.fluid_downsample_operator(x0_hat, scale_factor=self.args.r)
            loss = torch.norm(y - y_hat, p=2) ** 2

            guidance_grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
            

            with torch.no_grad():

                x_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred)
                
                # Apply Dynamic Step-Size scaling as recommended by the authors
                norm_factor = torch.norm(y - y_hat, p=2)
                dynamic_zeta = zeta / (norm_factor + 1e-8)
                
                # Inject the DPS Guidance nudge directly into the trajectory
                x_mean = x_mean - dynamic_zeta * guidance_grad
                
                # Add random SDE noise (dw -> z) if not at the final step
                if i > 0:
                    # Calculate variance according to your schedule
                    variance = beta_t * (1.0 - self.alphas_bar[i-1]) / (1.0 - alpha_bar_t)
                    noise = torch.randn_like(x)
                    x = x_mean + torch.sqrt(variance) * noise
                else:
                    x = x_mean
                    
                # Clean detach to prepare for the next loop's gradient tracking
                x = x.detach()
                
        return x