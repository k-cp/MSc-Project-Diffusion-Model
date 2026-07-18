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
    args = parser.parse_args() # Tell Python to check for rules set inside parser

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f: # Opens YAML file (set by args.config) in read mode
        config = yaml.safe_load(f) # Parse YAML file into python dict
    config = dict2namespace(config) # Convert dict to namesapce

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
