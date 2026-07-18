"""Train a Stochastic Interpolant drift network for flow super-resolution.

Trains b_theta on paired (x0=low-res, x1=high-res) standardized vorticity
stacks using the drift-matching objective (Eq 9 of Schiodt et al. 2026), then
saves the weights to --si_ckpt. Use main.py --run_si to super-resolve with the
trained checkpoint.

Example:
    python train_si.py --config kmflow_re1000_rs256_conditional.yml \
        --epochs 2000 --batch_size 32 --lr 2e-4 --frame_stride 4 \
        --si_ckpt ./pretrained_weights/si_ckpt.pth
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import yaml

from runners.rs256_guided_diffusion import StdScaler
from runners.stochastic_interpolant import (
    StochasticInterpolant,
    load_si_pairs,
    load_si_targets,
    DegradationSampler,
    SIPairsDataset,
    si_worker_init_fn,
)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def parse_args():
    p = argparse.ArgumentParser(description="Train a stochastic-interpolant drift network")
    p.add_argument("--config", type=str, required=True, help="Config file in configs/")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=50)
    p.add_argument("--frame_stride", type=int, default=4,
                   help="Subsample frames within each trajectory (decorrelate + save memory)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--si_ckpt", type=str, default="./pretrained_weights/si_ckpt.pth",
                   help="Output path for the trained drift network")
    p.add_argument("--resume", type=int, default=0,
                   help="Set to 1 to resume from --si_ckpt if it exists (continue epoch/optimizer)")
    p.add_argument("--si_physics", type=str, default="none", choices=["none", "learned"],
                   help="'learned' trains the drift net with the PDE residual gradient as an extra "
                        "conditioning input (Shu et al. Alg 1). 'linear' guidance needs NO special "
                        "training -- train with 'none' and enable it at inference.")
    p.add_argument("--si_pu", type=float, default=0.1,
                   help="Probability of dropping the physics condition during 'learned' training "
                        "(enables classifier-free guidance at sampling)")
    # --- blind SI (Option 1): manufacture x0 on the fly with random degradations ---
    p.add_argument("--si_augment", type=int, default=0,
                   help="1 = 'blind' training: ignore the fixed u3232 and generate x0 from the "
                        "ground truth each step with a randomly sampled degradation (robust to "
                        "different low-res inputs). 0 = train on the fixed u3232 sensor field.")
    p.add_argument("--si_aug_families", type=str, default="sensor",
                   help="Comma list of degradation families for --si_augment: "
                        "sensor,downsample,lowpass")
    p.add_argument("--si_aug_nmin", type=int, default=256,
                   help="Min sensor count for the 'sensor' family (random per sample)")
    p.add_argument("--si_aug_nmax", type=int, default=4000,
                   help="Max sensor count for the 'sensor' family (random per sample)")
    p.add_argument("--si_aug_noise", type=float, default=0.0,
                   help="Std of additive measurement noise on x0 (standardized units)")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("LOG")

    with open(os.path.join("configs", args.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    worker_init = None
    if args.si_augment:
        # Blind SI (Option 1): load only the high-res targets; a fresh random
        # degradation makes x0 on the fly (in the workers), so x0 is never frozen.
        families = tuple(f.strip() for f in args.si_aug_families.split(",") if f.strip())
        logger.info(f"BLIND training: manufacturing x0 on the fly. families={families} "
                    f"sensor N in [{args.si_aug_nmin},{args.si_aug_nmax}] noise={args.si_aug_noise}")
        x1, data_mean, data_std = load_si_targets(
            config.data.data_dir, split="train", frame_stride=args.frame_stride,
        )
        scaler = StdScaler(data_mean, data_std)
        logger.info(f"Train targets: {x1.shape[0]} (frame_stride={args.frame_stride}), "
                    f"mean={data_mean:.4f} std={data_std:.4f}")
        degrader = DegradationSampler(
            families=families, n_min=args.si_aug_nmin, n_max=args.si_aug_nmax,
            meas_noise=args.si_aug_noise,
        )
        dataset = SIPairsDataset(x1, degrader)
        n_workers = max(4, args.num_workers)     # overlap the CPU degradation with the GPU
        worker_init = si_worker_init_fn
    else:
        logger.info("Loading training pairs (x0=low-res, x1=high-res)...")
        x0, x1, data_mean, data_std = load_si_pairs(
            config.data.data_dir,
            config.data.sample_data_dir,
            config.data.data_kw,
            split="train",
            frame_stride=args.frame_stride,
        )
        scaler = StdScaler(data_mean, data_std)
        logger.info(f"Train pairs: {x0.shape[0]} (frame_stride={args.frame_stride}), "
                    f"mean={data_mean:.4f} std={data_std:.4f}")
        # Keep raw tensors; standardize per-batch to avoid a second full-size copy.
        dataset = torch.utils.data.TensorDataset(x0, x1)
        n_workers = args.num_workers

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=n_workers, drop_last=True, worker_init_fn=worker_init,
        persistent_workers=(n_workers > 0),
    )

    si = StochasticInterpolant(config, device, logger, physics=args.si_physics)
    si.model.train()
    if args.si_physics == "learned":
        logger.info(f"Training WITH learned physics conditioning (p_uncond={args.si_pu}). "
                    f"This checkpoint is only usable with --si_physics learned at inference.")
    optimizer = torch.optim.AdamW(si.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(epoch):
        # linear warmup, then cosine annealing to 0
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    os.makedirs(os.path.dirname(args.si_ckpt) or ".", exist_ok=True)

    # Resume from a previous checkpoint if requested (e.g. after a 24h timeout).
    start_epoch = 0
    if args.resume and os.path.exists(args.si_ckpt):
        ckpt = torch.load(args.si_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            si.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0))
            logger.info(f"Resumed from {args.si_ckpt} at epoch {start_epoch}")
        else:
            # Old inference-only format ([state_dict]); load weights, restart schedule.
            state = ckpt[-1] if isinstance(ckpt, (list, tuple)) else ckpt
            si.model.load_state_dict(state)
            logger.info(f"Loaded weights from {args.si_ckpt} (no epoch info); starting at epoch 0")
    elif args.resume:
        logger.info(f"--resume set but {args.si_ckpt} not found; starting fresh")

    def save_ckpt(epoch):
        torch.save({"epoch": epoch, "model": si.model.state_dict(),
                    "optimizer": optimizer.state_dict()}, args.si_ckpt)
        logger.info(f"Saved checkpoint (epoch {epoch}) -> {args.si_ckpt}")

    for epoch in range(start_epoch, args.epochs):
        scale = lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale

        running, n_batches = 0.0, 0
        for x0_b, x1_b in loader:
            x0_b = scaler(x0_b.to(device))
            x1_b = scaler(x1_b.to(device))

            optimizer.zero_grad()
            loss = si.interpolant_loss(x0_b, x1_b, scaler=scaler, p_uncond=args.si_pu)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        if epoch % args.log_every == 0:
            logger.info(f"epoch {epoch + 1}/{args.epochs}  loss={running / max(1, n_batches):.6f}  "
                        f"lr={args.lr * scale:.2e}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_ckpt(epoch + 1)

    logger.info("Training finished.")


if __name__ == "__main__":
    sys.exit(main())
