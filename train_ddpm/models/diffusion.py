import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
    # return torch.nn.GroupNorm(num_groups=20, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="circular")
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     padding_mode='circular')
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     padding_mode='circular')
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
   # =============================================================================
    #  ConditionalModel — DDPM U-Net for 2D Kolmogorov flow
    #  config: kmflow_re1000_rs256_conditional.yml
    #    ch=64, ch_mult=[1,1,1,2], num_res_blocks=1, dropout=0.0, resamp_with_conv=True
    #    attn_resolutions=[16] matches no level (256/128/64/32) -> inert
    #    3.5M params. All 3x3 convs use circular padding (periodic domain).
    #    14 ResnetBlocks, 1 AttnBlock (bottleneck only).
    # =============================================================================
    #
    # =============================================================================
    #  PART 1 — DATA FLOW           what happens, in execution order
    # =============================================================================
    #
    # --- inputs ------------------------------------------------------------------
    #   x    noisy vorticity, 3 consecutive frames        (3, 256, 256)
    #   t    diffusion timestep, one int per batch item   scalar
    #   dx   Navier-Stokes residual gradient              (3, 256, 256) or None
    #        the loss passes dx=None 10% of the time (p=0.1), so one set of weights
    #        serves both conditional and unconditional sampling
    #
    # --- timestep branch ---------------------------------------------------------
    #   runs first in forward(), but is INDEPENDENT of x — parallel, not sequential
    #   t  -> get_timestep_embedding(t, 64)                    ->  (64,)
    #      -> temb.dense[0] -> swish -> temb.dense[1]          ->  (256,)
    #   result is injected inside every ResnetBlock; nowhere else
    #
    # --- stem: lift to feature space, fuse conditioning --------------------------
    #   x   -> conv_in                                         ->  (64, 256, 256)
    #   dx  -> emb_conv        (zeros_like if dx is None)      ->  (64, 256, 256)
    #          torch.cat along channels                        ->  (128, 256, 256)
    #          combine_conv (1x1)                              ->  (64, 256, 256)  push skip[0]
    #
    # --- encoder: resblock and downsample strictly alternate ---------------------
    #   L0   resblock    64 -> 64                              ->  (64, 256, 256)  push skip[1]
    #        downsample  256 -> 128                            ->  (64, 128, 128)  push skip[2]
    #   L1   resblock    64 -> 64                              ->  (64, 128, 128)  push skip[3]
    #        downsample  128 -> 64                             ->  (64,  64,  64)  push skip[4]
    #   L2   resblock    64 -> 64                              ->  (64,  64,  64)  push skip[5]
    #        downsample  64 -> 32                              ->  (64,  32,  32)  push skip[6]
    #   L3   resblock    64 -> 128   (ch_mult[3] = 2)          ->  (128, 32,  32)  push skip[7]
    #        no downsample (last level)
    #   resblocks change channels only; downsamples change resolution only
    #
    # --- bottleneck: hard-coded, not driven by any config field ------------------
    #        mid.block_1   resblock                            ->  (128, 32, 32)
    #        mid.attn_1    self-attention over 1024 positions  ->  (128, 32, 32)
    #        mid.block_2   resblock                            ->  (128, 32, 32)
    #
    # --- decoder: 2 resblocks per level (num_res_blocks+1), each pops one skip ---
    #   L3   cat(h, skip[7])   (256, 32,  32)  -> resblock     ->  (128, 32,  32)
    #        cat(h, skip[6])   (192, 32,  32)  -> resblock     ->  (128, 32,  32)
    #        upsample    32 -> 64                              ->  (128, 64,  64)
    #   L2   cat(h, skip[5])   (192, 64,  64)  -> resblock     ->  (64,  64,  64)
    #        cat(h, skip[4])   (128, 64,  64)  -> resblock     ->  (64,  64,  64)
    #        upsample    64 -> 128                             ->  (64, 128, 128)
    #   L1   cat(h, skip[3])   (128, 128, 128) -> resblock     ->  (64, 128, 128)
    #        cat(h, skip[2])   (128, 128, 128) -> resblock     ->  (64, 128, 128)
    #        upsample    128 -> 256                            ->  (64, 256, 256)
    #   L0   cat(h, skip[1])   (128, 256, 256) -> resblock     ->  (64, 256, 256)
    #        cat(h, skip[0])   (128, 256, 256) -> resblock     ->  (64, 256, 256)
    #        no upsample (top level); skip stack now empty
    #
    # --- output ------------------------------------------------------------------
    #        norm_out -> swish -> conv_out                     ->  (3, 256, 256)
    #   this is the predicted NOISE e, NOT the reconstructed flow field
    #
    #
    # =============================================================================
    #  PART 2 — BLOCK REFERENCE     how each piece works internally
    #                               (nothing below is a step in the flow)
    # =============================================================================
    #
    # --- ResnetBlock -------------------------------------------------------------
    #   Computes an update h from x AND the timestep, returns x + h.
    #   Spatial size always preserved — that is what makes the addition valid.
    #
    #     u = swish( GN1(x) )
    #     a = W1 * u                      first 3x3 circular conv
    #     b = a + Wt @ swish(tau)         timestep: one value per channel,
    #                                     broadcast identically to all H*W pixels
    #     v = dropout( swish( GN2(b) ) )  dropout is a no-op here (p = 0.0)
    #     h = W2 * v                      second 3x3 circular conv
    #     y = x + h
    #
    #   Two stacked 3x3 convs => each value of h depends on a 5x5 patch of x,
    #   across all channels.
    #   Exception: when in_channels != out_channels the shortcut must be reshaped,
    #   y = nin_shortcut(x) + h.  This applies to 9 of the 14 blocks: down.3 (64->128)
    #   and all 8 decoder blocks (wide on input because of skip concatenation).
    #
    # --- Downsample --------------------------------------------------------------
    #   The same convolution as above, but the window steps 2 pixels instead of 1.
    #
    #     y[c,i,j] = SUM_c' SUM_a SUM_b  W[c,c',a,b] * x[c', 2i+a, 2j+b] + bias[c]
    #                                    a,b in {0,1,2}
    #
    #   - the 3x3 window is SPATIAL; it spans ALL input channels
    #     -> 64 channels * 9 positions = 576 terms per output value
    #   - 64 separate kernels -> 64 output channels; channel count unchanged
    #   - circular pad first (256 -> 257) so the last window wraps to the far edge
    #   - the 2i, 2j is the whole mechanism: output GRID becomes 1/4 the size
    #   - NOTHING is discarded — windows overlap by one column, so every input
    #     pixel is read by at least one window (verified: 0 of 256 unused)
    #
    # --- Upsample ----------------------------------------------------------------
    #   nearest-neighbour x2, then a 3x3 circular conv to smooth the blockiness.
    #   (the conv is present because resamp_with_conv=True; without it, bare
    #   interpolation would leave 2x2 constant patches)
    #   Changes resolution only; channel count unchanged.
    #
    # --- AttnBlock ---------------------------------------------------------------
    #   Computes an update h from x ALONE — no timestep — and returns x + h.
    #   Shape preserved, same as ResnetBlock. Runs once, as mid.attn_1.
    #
    #     z = GN(x)
    #     Q = Wq * z,  K = Wk * z,  V = Wv * z    three 1x1 convs
    #                                             (per-pixel maps, no neighbour mixing)
    #     flatten space: each becomes 1024 vectors of length 128  (32*32 = 1024)
    #
    #     A[i,j] = softmax_j( Q[i].K[j] / sqrt(C) )   how much position j matters to i
    #     out[j] = SUM_i  V[i] * A[i,j]               weighted average over ALL positions
    #
    #     h = Wo * out
    #     y = x + h
    #
    #   Difference from ResnetBlock: h is built from ALL 1024 positions, not a local
    #   5x5 patch — and A is computed from the input at runtime, not fixed by training.
    #   Cost is (H*W)^2: ~1M entries at 32x32, but ~4.3 BILLION at 256x256, which is
    #   why attention only ever appears at the bottleneck.
    # =============================================================================
    def __init__(self, config): # Building parts
        super().__init__()
        self.config = config

         # Pull config from YAML file
         # ch: number of channels after　Convolutional Layer : 64
         # out_ch:  number of output channel. 3 for number of simulation snapshots in consecutive times (a window)
         # ch_mult: shape of the U-Net's descent

        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)


        num_res_blocks = config.model.num_res_blocks # how many ResnetBlocks per resolution LEVEL
        attn_resolutions = config.model.attn_resolutions # list of spatial resolutions at which to insert an AttnBlock
        dropout = config.model.dropout # fraction of activations randomly zeroed during training (off at eval)
        in_channels = config.model.in_channels # channel count of the INPUT DATA = 3 consecutive vorticity frames
        resolution = config.data.image_size # expected input height = width
        resamp_with_conv = config.model.resamp_with_conv 
        # learned resizing (True) vs fixed arithmetic (False).
        #   False: downsample = avg_pool2d(2,2) (slides window 2*2 and get average) upsample = nearest x2 only (copies each value to a 2×2 block to make whole bigger by 4* )
        #   True : downsample = stride-2 conv  （slides 3*3 window, see below explanation)  upsample = nearest x2 + 3x3 conv (every cell is replaced by a weighted sum of its 3×3 neighbourhood)
        num_timesteps = config.diffusion.num_diffusion_timesteps # length of the diffusion noise schedule (1000 steps)
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps)) # create one learnable log-variance per timestep (1000 scalars)

        # Store sizes 
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([ # put a 2-item list inside the container
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       padding_mode='circular')
        curr_res = resolution # tracks current spatial size, starts at 256
        in_ch_mult = (1,)+ch_mult # Shift ch_mult by one
        self.down = nn.ModuleList() # Empty container
        block_in = None # tracks the running channel width during the loop 
        for i_level in range(self.num_resolutions): # runs 4 times: 0, 1, 2, 3
            block = nn.ModuleList() # Collect this level's resblock
            attn = nn.ModuleList() # Collect this level's attention blocks
            block_in = ch*in_ch_mult[i_level] # input channels for this level
            block_out = ch*ch_mult[i_level]  # output channels for this level
            for i_block in range(self.num_res_blocks): # just 1 block per level here for cond config
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            # package the level and register it
            down = nn.Module() # Create local down
            down.block = block # Attach resblock list
            down.attn = attn # Attach attention list
            if i_level != self.num_resolutions-1: # every level except the last
                down.downsample = Downsample(block_in, resamp_with_conv) # attach a downsampler
                curr_res = curr_res // 2 # halves resolution
            self.down.append(down) # hand it to the ModuleList

        # middle 
        self.mid = nn.Module() 
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) 
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # 3,2,1,0 <— deepest first
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1): # one EXTRA block per level
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in, # ADD widths 
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module() # create local up
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in) # Standardize number
        self.conv_out = torch.nn.Conv2d(block_in, # Squeeze 64 channels into 3 (nb of snapshots)
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution # Reject anything that doesn't have adequate resolution

        # timestep embedding
        # t = n          ->  64 numbers  ->  256  ->  256  ->  256
        #                     (line 1)       (l.2)   (l.3)   (l.4)
        temb = get_timestep_embedding(t, self.ch) # Computes sine and cosine of t at 32 different speeds (t = angle / freq), giving 64 numbers

        # print("#### models -> diffusion -> Model -> forward ####")
        # print("t.size(): ", t.size())
        # print("self.ch: ", self.ch)
        # print("temb.size(): ", temb.size())

        temb = self.temb.dense[0](temb) # Linear(64 -> 256)   widen and mix
        temb = nonlinearity(temb) # swish, elementwise (x = x * sigmoid(z))
        temb = self.temb.dense[1](temb) # Linear(256 -> 256)  refine

        # downsampling
        hs = [self.conv_in(x)]
        # print("x.size(): ", x.size())
        # print("hs[-1].size(): ", hs[-1].size())
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # if i_level == 0:
                #     print("i_level: ", i_level)
                #     print("hs[-1].size(): ", hs[-1].size())
                #     print("temb.size(): ", temb.size())
                #     print("h.size(): ", h.size())
                #     print("self.down[i_level].block[i_block]\n", self.down[i_level].block[i_block])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class ConditionalModel(nn.Module):
    """DDPM U-Net for 2D Kolmogorov flow, conditioned on a physics-residual field.

    Identical to `Model` above except for three additions: the `emb_conv` branch
    that embeds `dx`, the `combine_conv` that fuses it into the main path, and a
    circular `padding_mode` on `conv_out` (which `Model` omits). +45,440 params.

    Shapes below are for configs/kmflow_re1000_rs256_conditional.yml:
        ch=64, ch_mult=[1,1,1,2], num_res_blocks=1, dropout=0.0,
        attn_resolutions=[16], image_size=256  ->  3,514,179 params.

    ---- inputs ----------------------------------------------------------------
      x    noisy vorticity, 3 consecutive frames        (3, 256, 256)
      t    diffusion timestep, one int per batch item   scalar
      dx   grad of the Navier-Stokes residual           (3, 256, 256) or None
           conditional_noise_estimation_loss passes dx=None 10% of the time
           (p=0.1) so one set of weights serves conditional and unconditional
           sampling.  NOTE: voriticity_residual returns dx[0] == -dx[2] exactly
           (frames 0 and 2 enter the residual only via the central-difference
           dw/dt), so only 2 of the 3 conditioning channels are independent.

    ---- timestep branch -------------------------------------------------------
      Computed first in forward() but INDEPENDENT of x -- parallel, not
      sequential.  Injected by broadcast addition inside all 14 ResnetBlocks;
      AttnBlock does not receive it.
        t -> get_timestep_embedding(t, 64)     (64,)   fixed sinusoids
          -> dense[0] -> swish -> dense[1]     (256,)  learned

    ---- stem ------------------------------------------------------------------
      x  -> conv_in                            (64, 256, 256)
      dx -> emb_conv   (zeros_like if None)    (64, 256, 256)
            cat along channels                 (128, 256, 256)
            combine_conv (1x1)                 (64, 256, 256)   push skip[0]

    ---- encoder: resblock and downsample strictly alternate -------------------
      L0  resblock  64 -> 64                   (64, 256, 256)   push skip[1]
          downsample                           (64, 128, 128)   push skip[2]
      L1  resblock  64 -> 64                   (64, 128, 128)   push skip[3]
          downsample                           (64,  64,  64)   push skip[4]
      L2  resblock  64 -> 64                   (64,  64,  64)   push skip[5]
          downsample                           (64,  32,  32)   push skip[6]
      L3  resblock  64 -> 128 (ch_mult[3]=2)   (128, 32,  32)   push skip[7]
          no downsample (last level)
      ResnetBlocks change channels only; Downsample changes resolution only.

    ---- bottleneck: hard-coded, not driven by any config field ----------------
      mid.block_1 -> mid.attn_1 -> mid.block_2 (128, 32, 32)

    ---- decoder: num_res_blocks+1 = 2 resblocks per level, one skip each ------
      L3  cat(h, skip[7]) (256,32,32)  -> resblock   (128, 32,  32)
          cat(h, skip[6]) (192,32,32)  -> resblock   (128, 32,  32)
          upsample                                   (128, 64,  64)
      L2  cat(h, skip[5]) (192,64,64)  -> resblock   (64,  64,  64)
          cat(h, skip[4]) (128,64,64)  -> resblock   (64,  64,  64)
          upsample                                   (64, 128, 128)
      L1  cat(h, skip[3]) (128,...)    -> resblock   (64, 128, 128)
          cat(h, skip[2]) (128,...)    -> resblock   (64, 128, 128)
          upsample                                   (64, 256, 256)
      L0  cat(h, skip[1]) (128,...)    -> resblock   (64, 256, 256)
          cat(h, skip[0]) (128,...)    -> resblock   (64, 256, 256)
          no upsample (top level); skip stack now empty

    ---- output ----------------------------------------------------------------
      norm_out -> swish -> conv_out              (3, 256, 256)
      This is the predicted NOISE e, not the reconstructed flow field --
      losses.py compares it against the e that was added to x0.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # gradient embedding
        # 3-stage network that processes dx into the same shape as conv_in(x)
        # [0] Conv2d(3, 64, kernel_size=1)          -> (64, 256, 256)      256 params
        # [1] GELU                                  -> (64, 256, 256)        0 params
        # [2] Conv2d(64, 64, kernel_size=3, circ)   -> (64, 256, 256)   36,928 params
        #                                                        total  37,184
        self.emb_conv = nn.Sequential(
            torch.nn.Conv2d(in_channels, self.ch, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            torch.nn.Conv2d(self.ch, self.ch, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        )

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                        padding=1, padding_mode='circular')

            

        self.combine_conv = torch.nn.Conv2d(self.ch*2, self.ch, kernel_size=1, stride=1, padding=0)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular')
        # self.spectral_regressor1 = SpectralConv2d_fast(ch, ch, 12, 12)
        # self.spectral_regressor2 = SpectralConv2d_fast(ch, ch, 12, 12)
        # self.final_norm = Normalize(ch)
        # self.to_out = torch.nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x, t, dx=None):
        """Predict the noise that was added to a clean vorticity field.

        Args:
            x:  (B, 3, 256, 256) noisy vorticity, 3 consecutive frames.
            t:  (B,) diffusion timestep per batch element, 0..999.
            dx: (B, 3, 256, 256) gradient of the Navier-Stokes residual, or
                None. When None the conditioning contributes zeros, so the same
                weights run unconditionally.

        Returns:
            (B, 3, 256, 256) -- the predicted NOISE e, NOT the flow field.
            losses.py compares this against the e that was added to x0.

        Steps:
            1. assert x is square and matches self.resolution
            2. encode t: sinusoids (64) -> MLP (256); read by every ResnetBlock
            3. conv_in lifts x           3 -> 64 channels
            4. emb_conv embeds dx        3 -> 64 channels (zeros if dx is None)
            5. cat -> 128, combine_conv -> 64, seed the skip stack
            6. encoder: 4 levels of resblock + downsample; 256->32 px,
               64->128 ch; pushes 7 more skips (8 total)
            7. bottleneck: resblock -> attention -> resblock, all (128, 32, 32)
            8. decoder: 4 levels, 2 skip-pops each; 32->256 px, 128->64 ch;
               stack ends empty
            9. norm_out -> swish -> conv_out, 64 -> 3

        Notes:
            Deterministic (dropout is 0.0), and batch elements never interact --
            every norm here is GroupNorm, whose statistics are per-sample, so
            batch size does not affect the result.
            Builds nothing: every layer used was constructed in __init__.
        """
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)


        x = self.conv_in(x)
        if dx is not None:
            cond_emb = self.emb_conv(dx)
        else:
            cond_emb = torch.zeros_like(x)
        x = torch.cat((x, cond_emb), dim=1)
    
        # downsampling
        hs = [self.combine_conv(x)] # 

        for i_level in range(self.num_resolutions): # 0, 1, 2, 3
            for i_block in range(self.num_res_blocks): # once (num_res_blocks=1)
                h = self.down[i_level].block[i_block](hs[-1], temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
