import os
import numpy as np

from functions import dead_block
from tqdm import tqdm

import torch
import torchvision.utils as tvu
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.diffusion_new import ConditionalModel as CModel
from models.diffusion_new import Model
from functions.process_data import *
from functions.denoising_step import guided_ddpm_steps, guided_ddim_steps, ddpm_steps, ddim_steps

import matplotlib.pyplot as plt
from einops import rearrange
from mpl_toolkits.axes_grid1 import ImageGrid
import math
import pickle

from copy import deepcopy


class MetricLogger(object):
    def __init__(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        self.metric_dict = {}
        self.reset()

    def reset(self):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key] = []

    @torch.no_grad()
    def update(self, **kwargs):
        for key in self.metric_fn_dict.keys():
            self.metric_dict[key].append(self.metric_fn_dict[key](**kwargs))

    def get(self):
        return self.metric_dict.copy()

    def log(self, outdir, postfix=''):
        with open(os.path.join(outdir, f'metric_log_{postfix}.pkl'), 'wb') as f:
            pickle.dump(self.metric_dict, f)


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def load_flow_data(path, stat_path=None):
    # load flow data from path
    data = np.load(path)   # [N, T, h, w]

    print('Original data shape:', data.shape)
    data_mean, data_scale = np.mean(data[:-4]), np.std(data[:-4])
    print(f'Data range: mean: {data_mean} scale: {data_scale}')
    data = data[-4:, ...].copy().astype(np.float32)   # only take the test set
    data = torch.as_tensor(data, dtype=torch.float32)
    flattened_data = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-2):
            flattened_data.append(data[i, j:j+3, ...])
    flattened_data = torch.stack(flattened_data, dim=0)
    print(f'data shape: {flattened_data.shape}')
    return flattened_data, data_mean.item(), data_scale.item()


def load_recons_data(ref_path, sample_path, data_kw, smoothing, smoothing_scale):
    with np.load(sample_path, allow_pickle=True) as f:
        sampled_data = f[data_kw][-4:, ...].copy().astype(np.float32) 
        # u3232 has dim (40, 320, 256, 256), extract last 4 (test set): (4, 320, 256, 256)
        #(number of batches, total frames in batch, height, width)

        # idx_lst = f[idx_kw][-4:]
    sampled_data = torch.as_tensor(sampled_data, dtype=torch.float32)
    ref_data = np.load(ref_path).astype(np.float32)
    data_mean, data_scale = np.mean(ref_data[:-4]), np.std(ref_data[:-4])
    ref_data = ref_data[-4:, ...].copy().astype(np.float32)   # only take the test set
    ref_data = torch.as_tensor(ref_data, dtype=torch.float32)

    flattened_sampled_data = []
    flattened_ref_data = []

    for i in range(ref_data.shape[0]):  # data restrucuturing
        for j in range(ref_data.shape[1] - 2):
            flattened_ref_data.append(ref_data[i, j:j + 3, ...])
            flattened_sampled_data.append(sampled_data[i, j:j + 3, ...])
            # mask_lst.append(mask)
    # flattened_ref_data and flattened_sampled_data are a list of 1272 (3, 256, 256)

    flattened_ref_data = torch.stack(flattened_ref_data, dim=0) # Compress list into single tensor of dim (1272, 3, 256, 256)
    flattened_sampled_data = torch.stack(flattened_sampled_data, dim=0) # Compress list into single tensor of dim (1272, 3, 256, 256)
    if smoothing: # Apply Gaussian blur
        arr = flattened_sampled_data
        ker_size = smoothing_scale
        # peridoic padding
        arr = F.pad(arr,
                    pad=((ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2, (ker_size - 1) // 2),
                    mode='circular', )
        arr = transforms.GaussianBlur(kernel_size=ker_size, sigma=ker_size)(arr)# F.avg_pool2d(arr, (ker_size, ker_size), stride=1, count_include_pad=False)
        flattened_sampled_data = arr[..., (ker_size - 1) // 2:-(ker_size - 1) // 2, (ker_size - 1) // 2:-(ker_size - 1) // 2]

    print(f'data shape: {flattened_ref_data.shape}')

    return flattened_ref_data, flattened_sampled_data, data_mean.item(), data_scale.item()


class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        return (x - self.min) #/ (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def scale(self):
        return self.max - self.min


class StdScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def scale(self):
        return self.std


def nearest_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = data[:, :, ::scale, ::scale]

    return blur_data


def gaussian_blur_image(data, scale):
    # use pytorch's bulit-in transforms to blur the data
    blur_data = transforms.GaussianBlur(kernel_size=scale, sigma=2*scale+1)(data)

    return blur_data


def random_square_hole_mask(data, hole_size):
    # generate a random square hole mask
    h, w = data.shape[2:]
    mask = torch.zeros(data.shape, dtype=torch.int64).to(data.device)
    hole_x = np.random.randint(0, w - hole_size)
    hole_y = np.random.randint(0, h - hole_size)
    mask[..., hole_y:hole_y+hole_size, hole_x:hole_x+hole_size] = 1

    return mask


def make_image_grid(images, out_path, ncols=10):
    # assume images in the shape of (N, T, H, W)
    t, h, w = images.shape
    images = images.detach().cpu().numpy()
    b = t // ncols
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b, ncols),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b*ncols)):
        # Iterating over the grid returns the Axes.
        ax.imshow(images[im_no, :, :], cmap='twilight', vmin=-23, vmax=23)
        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_measurement_noise(x, sigma, seed, batch_index):
    """Corrupt a measurement with Gaussian sensor noise, in STANDARDIZED units.

    Shared by the baseline, DPS and SI runners so all three are degraded the
    same way. Seeded per (seed, batch_index) so a run is reproducible and the
    SAME noise realisation is used across methods for a fair comparison.

    sigma is relative to the standardized field (std 1), so 0.02 == 2% of the
    field's spread. Returns x unchanged when sigma == 0.
    """
    if not sigma:
        return x
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 100003 + int(batch_index))
    noise = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device)
    return x + sigma * noise


def slice2sequence(data): # Remove overlapping 
    data = rearrange(data[:, 1:2], 't f h w -> (t f) h w')
    return data


def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def l2_loss(x, y):
    return ((x - y)**2).mean((-1, -2)).sqrt().mean()


def voriticity_residual(w, re=1000.0, dt=1/32, calc_grad=True):
    # w [b t h w] Vorticity
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2) # Spatial resolution: x size
    ny = w.size(3) # Spatial resolution: y size
    device = w.device

    # Extract central frame from 3 frame: w[:, 1:-1] (20, 1, 256, 256)
    # Run 2D Fast Fourier Transform to turn to complex frequency space
    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3]) 

    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx

    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), # frequenceis = [0, 1, 2, ... , 127, -128, -127, ... , -1]
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)  # Replcate column to to form N * N and change into 4dtensor of (1, 1, 256, 256)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), # frequenceis = [0, 1, 2, ... , 127, -128, -127, ... , -1]
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)   # Replcate column to to form N * N and change into 4dtensor of (1, 1, 256, 256)
    
    lap = (k_x ** 2 + k_y ** 2) # Lacplacian operator 
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap # Streamfunction

    u_h = 1j * k_y * psi_h # Horizontal velocity: u
    v_h = -1j * k_x * psi_h # Vertical velocity: v

    wx_h = 1j * k_x * w_h # Horizontal gradient of vorticity: d ω / d x 
    wy_h = 1j * k_y * w_h # Vertical gradient of vorticity: d ω / d y

    wlap_h = -lap * w_h# Full physical diffusion term: -∇^2 ω

    # Inverse Real Fast Fourier Transform: convert back to real-numbered grid
    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])

    advection = u*wx + v*wy # u・∇ω = u * d ω / d x  + u * d ω / d y 

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt) # Time derivative: dω / dt (Take difference between frame 2 and 0)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device) # inisialise coordinate line from $0$ to $2\pi$ split into 256 steps
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x) # create a full 2D grid of physical coordinates 256 * 256
    f = -4*torch.cos(4*Y) # Create a steady, sinusoidal force field that pushes horizontally across the Y-axis

    residual = wt + (advection - (1.0 / re) * wlap + 0.1*w[:, 1:-1]) - f # Navier Stokes equation for vorticity transport rearranged to one side 
    # (20, 1, 256, 256) (calculated independently for all 20 batch sequences)

    residual_loss = (residual**2).mean() # Calculate residual loss for the entire batch of 20 sequences.

    if calc_grad: # Physics guidance
        dw = torch.autograd.grad(residual_loss, w)[0] # dw= d(residual loss) / dw
        return dw, residual_loss
    else: # Evaluation
        return residual_loss


class Diffusion(object):
    def __init__(self, args, config, logger, log_dir, device=None):
        self.args = args
        self.config = config
        self.logger = logger
        self.image_sample_dir = log_dir

        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule( # Create sequnce of scaling factors
            beta_start=config.diffusion.beta_start, 
            beta_end=config.diffusion.beta_end, 
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps # Total number of discrete timesteps 
        )
        self.betas = torch.from_numpy(betas).float().to(self.device) # Convert betas (numpy array) to tensor and moves it to device
        self.num_timesteps = betas.shape[0] 

        alphas = 1.0 - betas # Signal retention 
        alphas_cumprod = np.cumprod(alphas, axis=0) # Cummulative product of alpha: \bar α_t
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]) # Previous cummulative product of alpha: \bar α_t-1 = \bar α_t / α_t
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # Log of posterior variance 
        if self.model_var_type == "fixedlarge": 
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def log(self, info): # Log message inside log file
        self.logger.info(info)

    def reconstruct(self):
        self.log('Doing sparse reconstruction task')
        self.log("Loading model")

        if self.config.model.type == 'conditional':
            print('Using conditional model')
            model = CModel(self.config)
        else:
            print('Using unconditional model')
            model = Model(self.config)

        model.load_state_dict(torch.load(self.config.model.ckpt_path, map_location='cpu')[-1])

        model.to(self.device)

        self.log("Model loaded")

        model.eval()
        self.log('Preparing data')
        
        # Data preprocessing
        ref_data, blur_data, data_mean, data_std = load_recons_data(self.config.data.data_dir, # reference data: High resolution data (ground truth for the super-resolution task)
                                                                    self.config.data.sample_data_dir, # Low resolution data measured from random grid locations (input data for the super-resolution task) 
                                                                    self.config.data.data_kw, # which array inside the npz file to choose 
                                                                    smoothing=self.config.data.smoothing,
                                                                    smoothing_scale=self.config.data.smoothing_scale)

        # ref_data (1272, 3, 256, 256)
        # blur_data (1272, 3, 256, 256)

        _blk_blur, _ = dead_block.build_from_config(self.args, self.config,
                                                    blur_data, log=self.log)
        if _blk_blur is not None:
            blur_data = _blk_blur

        scaler = StdScaler(data_mean, data_std)
        # minmax_scaler = MinMaxScaler(ref_data.min(), ref_data.max())

        _bp = getattr(self.args, 'baseline_physics', 'both')
        self.log("Physics mechanism: {} ({})".format(_bp, {
            'both':   'learned encoding + direct gradient descent (repo default)',
            'cond':   'learned encoding ONLY -- dx conditions the net, not subtracted',
            'linear': 'direct gradient descent ONLY -- dx subtracted, unconditional net',
            'none':   'no physics guidance at all',
        }[_bp]))
        self.log("Start sampling")

        # pack data loader
        testset = torch.utils.data.TensorDataset(blur_data, ref_data) # Pair blurry data to reference data (ground truth)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=self.config.sampling.batch_size,  # Feed data in batch sizes instead of all at once (size = 20)
                                                  shuffle=False, num_workers=self.config.data.num_workers) # Number of paralel processes (4 by default in config files)
        # testset len() = 1272 pairs (blurry and ref) of matching data　(3, 256, 256)　tensor (frame, height, width)
        # test_loader len() = 64 batches of pairs of (20, 3, 256, 256) (batch size, frame, H, W)

        l2_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step)) # Initialise l2 loss btwn ground truth and blurry: ( , default=1, default=1)
        residual_loss_all = np.zeros((ref_data.shape[0], self.args.repeat_run, self.args.sample_step)) # Initialise physics residual loss matrix


        for batch_index, (blur_data, data) in enumerate(test_loader): 
            self.log('Batch: {} / Total batch {}'.format(batch_index, len(test_loader)))

            x0 = blur_data.to(self.device)  # Move blurry data to device
            # x0: a single batch with dim (20, 3, 256, 256) (batch size, frame, H, W) 20 different 3-frame sequence of blurry data

            gt = data.to(self.device) # Move high res (ground truth) data to device
            # gt: a single batch with dim (20, 3, 256, 256) (batch size, frame, H, W) 20 different 3-frame sequence of ref data
            gt = data.to(self.device) # Move high res (ground truth) data to device 

            self.log('Preparing reference image')
            self.log('Dumping visualization...')

            sample_folder = 'sample_batch{}'.format(batch_index) # Create folder name for each batch
            ensure_dir(os.path.join(self.image_sample_dir, sample_folder)) # Check if folder already exists

            sample_img_filename = 'input_image.png' # Name the file for the low-resolution or sparse visualization.
            path_to_dump = os.path.join(self.image_sample_dir, sample_folder, sample_img_filename) # Combine paths to save sample_img_filename inside sample_folder
            x0_masked = x0.clone() # Create seperate, idential copy of x0 (blurry) in GPU memory
            # print(torch.any(mask))
            # x0_masked[~mask] = np.nan

            # slice2sequence remove edges of each 3-frame sequence to prevent overlapping: dim: (20, 256, 256)
            # make_image_grid make an image with 20 images of each frame dim 256*256 for blurry data
            make_image_grid(slice2sequence(x0_masked), path_to_dump) 


            sample_img_filename = 'reference_image.png'
            path_to_dump = os.path.join(self.image_sample_dir, sample_folder, sample_img_filename)
            make_image_grid(slice2sequence(gt), path_to_dump) # Make an image with 20 images of each frame dim 256*256 for ref data
 
            # save as array
            if self.config.sampling.dump_arr: 
                np.save(os.path.join(self.image_sample_dir, sample_folder, 'input_arr.npy'),
                        slice2sequence(x0).cpu().numpy())
                np.save(os.path.join(self.image_sample_dir, sample_folder, 'reference_arr.npy'),
                        slice2sequence(data).cpu().numpy())

            # calculate initial loss
            #l1_loss_init = l1_loss(x0, gt)
            l2_loss_init = l2_loss(x0, gt)

            self.log('L2 loss init: {}'.format(l2_loss_init))
            gt_residual = voriticity_residual(gt)[1].detach() # Residual loss of reference data
            init_residual = voriticity_residual(x0)[1].detach() # Residual loss of blurry data
            self.log('Residual init: {}'.format(init_residual))
            self.log('Residual reference: {}'.format(gt_residual))

            x0 = scaler(x0) # Scale values
            # Simulated sensor noise (inference-time). Added AFTER scaling so
            # --meas_noise is in standardized units, matching --si_aug_noise.
            x0 = add_measurement_noise(x0, getattr(self.args, 'meas_noise', 0.0),
                                       self.args.seed, batch_index)
            xinit = x0.clone()
            
            # prepare loss function
            if self.config.sampling.log_loss:
                l2_loss_fn = lambda x: l2_loss(scaler.inverse(x).to(gt.device), gt) # L2 Loss function
                equation_loss_fn = lambda x: voriticity_residual(scaler.inverse(x),
                                                                 calc_grad=False) # Equation Loss function: residual loss of Navier Stokes

                logger = MetricLogger({ # Save functions
                    'l2 loss': l2_loss_fn,
                    'residual loss': equation_loss_fn
                })

            # we repeat the sampling for multiple times
            for repeat in range(self.args.repeat_run): # Repeat number of times as probabilistic
                self.log(f'Run No.{repeat}:')
                x0 = xinit.clone()
                for it in range(self.args.sample_step):  # we run the sampling for three times
                    
                    # Forward process
                    e = torch.randn_like(x0) # Tensor of randam gaussian noise with the same shape as x0
                    total_noise_levels = int(self.args.t * (0.7 ** it))

                    a = (1 - self.betas).cumprod(dim=0) # cumulative variance schedule array
                    x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt() # closed-form forward diffusion equation:

                    if self.config.model.type == 'conditional': # Create function to extract dw: gradient of the Navier-Stokes residual loss
                        physical_gradient_func = lambda x: voriticity_residual(scaler.inverse(x))[0] / scaler.scale()
                    elif self.config.sampling.lambda_ > 0: # Create function for physics guidance: irect gradient descent of physics-informed condition
                        physical_gradient_func = lambda x: \
                            voriticity_residual(scaler.inverse(x))[0] / scaler.scale() * self.config.sampling.lambda_

                    num_of_reverse_steps = int(self.args.reverse_steps * (0.7 ** it ))
                    betas = self.betas.to(self.device)

                    # DDIM allow faster computation by skipping
                    skip = total_noise_levels // num_of_reverse_steps
                    seq = range(0, total_noise_levels, skip) # sparse indexing sequence


                    # Executing sampling


                    # Physics ablation. Shu et al. give two mechanisms: LEARNED
                    # ENCODING (dx into the network) and DIRECT GRADIENT DESCENT
                    # (dx subtracted from the update). 'both' is this repo's
                    # historical default and reproduces the earlier results
                    # exactly; the others isolate each half.
                    bp = getattr(self.args, 'baseline_physics', 'both')
                    if self.config.model.type == 'conditional' and bp in ('both', 'cond'):
                        xs, _ = guided_ddim_steps(x, seq, model, betas,
                                                  w=self.config.sampling.guidance_weight,
                                                  dx_func=physical_gradient_func,
                                                  subtract_dx=(bp == 'both'),
                                                  cache=False, logger=logger)

                    elif bp == 'linear':  # direct gradient descent only (unconditional net)
                        xs, _ = ddim_steps(x, seq, model, betas,
                                           dx_func=physical_gradient_func, cache=False, logger=logger)
                    else:  # bp == 'none': no physics anywhere
                        xs, _ = ddim_steps(x, seq, model, betas, cache=False, logger=logger)

                    x = xs[-1] # Obtain final fully-denoised image
                    x0 = xs[-1].cuda()


                    # Measure L2 loss between ground truth and log it
                    l2_loss_f = l2_loss(scaler.inverse(x.clone()).to(gt.device), gt)
                    self.log('L2 loss it{}: {}'.format(it, l2_loss_f))

                    # Measure Navier Stokes euqation loss between ground truth and log it
                    residual_loss_f = voriticity_residual(scaler.inverse(x.clone()), calc_grad=False).detach()
                    self.log('Residual it{}: {}'.format(it, residual_loss_f))

                    # l2_loss_log[f'run_{repeat}'].append(l2_loss_f.item())
                    # residual_loss_log[f'run_{repeat}'].append(residual_loss_f.item())

                    # Store final losses.
                    # Offset MUST use the configured batch size, not x.shape[0]: the
                    # final batch is short (1272 = 63*20 + 12), so x.shape[0] would put
                    # it at row 63*12=756 instead of 63*20=1260 -- clobbering earlier
                    # rows and leaving the tail at zero (~1% bias in the reported mean).
                    _row0 = batch_index * self.config.sampling.batch_size
                    _row1 = _row0 + x.shape[0]
                    l2_loss_all[_row0:_row1, repeat, it] = l2_loss_f.item()
                    residual_loss_all[_row0:_row1, repeat, it] = residual_loss_f.item()

                    # Save the reconstruction as an image (same as input/reference).
                    sample_img_filename = f'sample_run_{repeat}_it{it}.png'
                    path_to_dump = os.path.join(self.image_sample_dir, sample_folder, sample_img_filename)
                    make_image_grid(slice2sequence(scaler.inverse(x)), path_to_dump)

                    if self.config.sampling.dump_arr: # Save raw data
                        np.save(os.path.join(self.image_sample_dir, sample_folder, f'sample_arr_run_{repeat}_it{it}.npy'),
                                slice2sequence(scaler.inverse(x)).cpu().numpy())

                    if self.config.sampling.log_loss: # Save log loss
                        logger.log(os.path.join(self.image_sample_dir, sample_folder), f'run_{repeat}_it{it}')
                        logger.reset()

            # with open(os.path.join(self.image_sample_dir, sample_folder, f'log.pkl'), 'wb') as f:
            #     pickle.dump({'l2 loss': l2_loss_log, 'residual loss': residual_loss_log, 'bicubic': bicubic_log}, f)
            self.log('Finished batch {}'.format(batch_index))
            self.log('========================================================')
        self.log('Finished sampling')
        self.log(f'mean l2 loss: {l2_loss_all[..., -1].mean()}')
        self.log(f'std l2 loss: {l2_loss_all[..., -1].std(axis=1).mean()}')
        self.log(f'mean residual loss: {residual_loss_all[..., -1].mean()}')
        self.log(f'std residual loss: {residual_loss_all[..., -1].std(axis=1).mean()}')



