import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy

from runners.rs256_guided_diffusion import Diffusion

torch.backends.mkldnn.enabled = False



def parse_args_and_config(): # Set default arguements
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file') # Must provide argument for --config
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--repeat_run', type=int, default=1, help='Repeat run')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--r', dest='reverse_steps', type=int, default=20, help='Revserse steps')
    parser.add_argument('--comment', type=str, default='', help='Comment')

    parser.add_argument("--sample_step", type=int, default=1, help="1 for standard repo sampling, 2 for custom DPS posterior sampling")
    parser.add_argument("--scale_factor", type=int, default=4, help="The downsampling factor for the fluid operator A(x)")
    parser.add_argument("--zeta", type=float, default=0.5, help="The gradient scale step-size for DPS guidance")
    parser.add_argument("--run_dps", type=int, default=0, help="Set to 1 to activate custom DPS posterior sampling")
    parser.add_argument("--operator", type=str, default="sparse", choices=["sparse", "downsample"],
                        help="DPS forward operator A: 'sparse' = 1024 random sensors from idx_lst (true measurement), "
                             "'downsample' = bicubic downsample by scale_factor (self-consistent SR smoke test)")
    parser.add_argument("--run_si", type=int, default=0,
                        help="Set to 1 to run Stochastic Interpolant super-resolution (needs a trained --si_ckpt)")
    parser.add_argument("--si_ckpt", type=str, default="./pretrained_weights/si_ckpt.pth",
                        help="Trained SI drift-network checkpoint (produced by train_si.py)")
    parser.add_argument("--si_steps", type=int, default=100,
                        help="Number of SDE integration steps for SI sampling")
    parser.add_argument("--si_physics", type=str, default="none",
                        choices=["none", "linear", "learned"],
                        help="Physics guidance for SI (Shu et al. JCP 2023): 'linear' = direct "
                             "gradient descent of the PDE residual (inference-only, use --si_lambda); "
                             "'learned' = PDE residual gradient as conditioning, classifier-free "
                             "(use --si_w; REQUIRES a checkpoint trained with --si_physics learned)")
    parser.add_argument("--si_lambda", type=float, default=0.0,
                        help="Step size for SI 'linear' physics guidance")
    parser.add_argument("--si_w", type=float, default=0.0,
                        help="Conditioning strength for SI 'learned' physics guidance")
    parser.add_argument("--si_tag", type=str, default="",
                        help="Extra suffix on the SI output folder (e.g. 'blind') so runs with "
                             "different trained checkpoints don't overwrite each other")
    parser.add_argument("--si_eval_degradation", type=str, default="",
                        help="Robustness eval: build x0 from a CHOSEN degradation of the ground "
                             "truth instead of the fixed u3232. Format 'family:param', e.g. "
                             "sensor:512, downsample:8, lowpass:4. Empty = use real u3232.")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Override the diffusion checkpoint in the config (model.ckpt_path). "
                             "Lets you run the SAME config against a different set of weights -- "
                             "e.g. the provided conditional_ckpt.pth vs one you trained yourself.")
    parser.add_argument("--baseline_tag", type=str, default="",
                        help="Extra suffix on the baseline/DPS output folder (e.g. 'mine') so a "
                             "self-trained checkpoint's output lands in its own folder instead of "
                             "overwriting the run made with the provided weights. Mirrors --si_tag.")
    parser.add_argument("--meas_noise", type=float, default=0.0,
                        help="INFERENCE-TIME sensor noise: std of Gaussian noise added to the "
                             "measurement each method consumes, in STANDARDIZED units (0.02 = 2%% "
                             "of the field's spread). This benchmark's sensors are exact "
                             "(u3232 == ground truth at sensor points), so nothing here ever "
                             "tests robustness to real instrument error -- which is precisely "
                             "the regime DPS was designed for. Applies to baseline, DPS and SI.")
    parser.add_argument("--dps_cond", type=int, default=0,
                        help="Feed the physics-informed condition into the NETWORK during DPS "
                             "(JCP's 'learned encoding'). DPS normally calls the conditional "
                             "checkpoint with dx=None, leaving the physics branch it was trained "
                             "with unused. Costs nothing and has no tunable strength -- unlike "
                             "--dps_physics, which adds a separate weighted force to the update.")
    parser.add_argument("--dps_physics", type=str, default="none",
                        choices=["none", "xt", "x0hat"],
                        help="Add a Navier-Stokes term to the DPS update. DPS enforces "
                             "MEASUREMENT consistency but no physics; the baseline does the "
                             "opposite. This fills that gap. 'xt' evaluates the residual on the "
                             "noisy field (what JCP does); 'x0hat' evaluates it on the Tweedie "
                             "clean estimate and chains the gradient back through the network "
                             "(DPS's own trick, applied to physics -- costs a 2nd backward pass).")
    parser.add_argument("--dps_lambda", type=float, default=1.0,
                        help="Strength of the DPS physics term (see --dps_physics). The log "
                             "prints both gradient norms on the first step so you can scale this.")
    parser.add_argument("--baseline_physics", type=str, default="both",
                        choices=["both", "cond", "linear", "none"],
                        help="Which physics mechanism the BASELINE uses. Shu et al. propose two "
                             "and this repo applies both by default. 'both'=learned encoding + "
                             "direct gradient descent (historical default); 'cond'=conditioning "
                             "only (dx fed to the net, not subtracted); 'linear'=direct gradient "
                             "descent only (dx subtracted, unconditional net); 'none'=no physics. "
                             "Ablates which half of the physics guidance actually does the work.")
    parser.add_argument("--guidance_weight", type=float, default=None,
                        help="Override sampling.guidance_weight from the config. This is the "
                             "classifier-free strength w in (w+1)*cond - w*uncond; the config "
                             "ships w=0, which disables the extrapolation. Overriding here beats "
                             "editing the YAML because that file is shared with the DPS/SI runs.")
    args = parser.parse_args() # Tell Python to check for rules set inside parser

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f: # Opens YAML file (set by args.config) in read mode
        config = yaml.safe_load(f) # Parse YAML file into python dict
    config = dict2namespace(config) # Convert dict to namesapce

    # Point the same config at different weights (provided vs self-trained) without
    # editing the YAML. Done before log_dir is written so the saved config.yml
    # records which checkpoint actually produced that folder.
    if getattr(args, 'ckpt_path', ''):
        config.model.ckpt_path = args.ckpt_path

    # Same idea for the classifier-free guidance strength. Applied before the
    # folder name is built, so a w-sweep automatically lands in _w<value>
    # folders instead of overwriting each other.
    if getattr(args, 'guidance_weight', None) is not None:
        config.sampling.guidance_weight = args.guidance_weight

    os.makedirs(config.log_dir, exist_ok=True) # Create folder named "log_dir"  (specified in YAML file) unless it exists
    if config.model.type == 'conditional': # Create direcotry name based on configurations

        dir_name = 'recons_{}_t{}_r{}_w{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.guidance_weight)
    else:

        dir_name = 'recons_{}_t{}_r{}_lam{}'.format(config.data.data_kw,
                                                    args.t, args.reverse_steps,
                                                    config.sampling.lambda_)

    if config.model.type == 'conditional': # Conditional diffusion: 
        print('Use residual gradient guidance during sampling')
        dir_name = 'guided_' + dir_name
    elif config.sampling.lambda_ > 0:
        print('Use residual gradient penalty during sampling')
        dir_name = 'pi_' + dir_name
    else:
        print('Not use physical gradient during sampling')

    # Keep DPS output in its own per-zeta folder so it never overwrites the
    # baseline (same t/r/w) AND so concurrent zeta-sweep jobs don't wipe each
    # other's sample_batch dirs (DPS clears each dir before writing).
    if getattr(args, 'run_dps', 0) == 1:
        dir_name = 'dps_' + dir_name + '_z{}'.format(args.zeta)
        # DPS + physics hybrid: tag by where the residual is evaluated and how
        # strongly it is applied, so a lambda sweep never overwrites itself.
        if getattr(args, 'dps_cond', 0) == 1:
            dir_name += '_cond'
        if getattr(args, 'dps_physics', 'none') != 'none':
            dir_name += '_phys{}_lam{}'.format(args.dps_physics, args.dps_lambda)

    # Stochastic Interpolant runs get their own si_ folder, tagged with the
    # physics-guidance mode and its strength so sweeps don't clobber each other.
    if getattr(args, 'run_si', 0) == 1:
        dir_name = 'si_' + dir_name
        si_physics = getattr(args, 'si_physics', 'none')
        if si_physics == 'linear':
            dir_name += '_linear_lam{}'.format(args.si_lambda)
        elif si_physics == 'learned':
            dir_name += '_learned_w{}'.format(args.si_w)
        # training-variant tag (e.g. 'blind') so a different checkpoint's output
        # lands in its own folder instead of overwriting the plain SI run.
        if getattr(args, 'si_tag', ''):
            dir_name += '_' + args.si_tag
        # robustness-eval tag: which degradation the model was fed (e.g. sensor:512
        # -> _eval_sensor512), so each held-out test gets its own folder.
        if getattr(args, 'si_eval_degradation', ''):
            dir_name += '_eval_' + args.si_eval_degradation.replace(':', '')

    # Which physics mechanism the baseline used. 'both' is the historical default,
    # so it stays untagged and existing folder names are unchanged.
    _bp = getattr(args, 'baseline_physics', 'both')
    if _bp != 'both' and getattr(args, 'run_si', 0) != 1 and getattr(args, 'run_dps', 0) != 1:
        dir_name += '_phys' + _bp

    # JCP's iterative-refinement rounds. It is NOT otherwise in the folder name,
    # so without this a --sample_step 3 run silently overwrites the 1-round
    # results. Only tagged when != 1, so existing folder names are unchanged.
    if getattr(args, 'sample_step', 1) != 1 and getattr(args, 'run_si', 0) != 1:
        dir_name += '_ss{}'.format(args.sample_step)

    # Which DIFFUSION checkpoint produced this run (baseline and DPS both use it).
    # SI has its own --si_tag, so this only tags the diffusion-driven modes: an
    # empty tag reproduces the original folder names, so runs made with the
    # provided weights stay exactly where they were.
    if getattr(args, 'baseline_tag', '') and getattr(args, 'run_si', 0) != 1:
        dir_name += '_' + args.baseline_tag

    # Inference-time sensor noise applies to EVERY method, so it is tagged last
    # and outside the per-method blocks. 0.0 leaves all existing names untouched.
    if getattr(args, 'meas_noise', 0.0):
        dir_name += '_mn{}'.format(args.meas_noise)

    log_dir = os.path.join(config.log_dir, dir_name) # Combine paths: config.log_dir/dir_name
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.yml'), 'w') as outfile: # Create full path to a new file named config.yml inside log_dir
        yaml.dump(config, outfile) # Save config in a YAML file inside outfile (file opened earlier) 

    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO) # Set threshold for logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Format log file output
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'logging_info')) # Create file and save it inside log_dir 
    file_handler.setLevel(logging.INFO) # Use INFO threshold created above
    file_handler.setFormatter(formatter) # Apply visual formatter created above
    logger.addHandler(file_handler) # Tell logger to log inside file_handler  

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device)) # Log device choice to file
    config.device = device # Save device choice inside config

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True # Choose best algorithm to perform optimisation

    return args, config, logger, log_dir # args: terminal inputs, config: YAML setting, logger: logging tool, log_dir: path to folder


def dict2namespace(config): # Convert dict to namesapce
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, logger, log_dir = parse_args_and_config()
    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid())) # Log process id in file
    logging.info("Exp comment = {}".format(args.comment)) # Log comment terminal input
    logging.info("Config =")
    print("<" * 80)

    try:

        if getattr(args, "run_si", 0) == 1:
            logging.info("Routing to Stochastic Interpolant super-resolution...")
            from runners.stochastic_interpolant import SIRunner
            runner = SIRunner(args, config, logger, log_dir)
            runner.si_sample_pipeline()
        elif getattr(args, "run_dps", 0) == 1:
            logging.info("Master Switch Activated: Routing to Diffusion Posterior Sampling (DPS)...")
            from runners.posterior_sampling import PosteriorRunner
            runner = PosteriorRunner(args, config, logger, log_dir)
            runner.dps_sample_pipeline()
        else:
            # DEFAULT REPOSITORY FLOW
            runner = Diffusion(args, config, logger, log_dir)
            runner.reconstruct()

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
