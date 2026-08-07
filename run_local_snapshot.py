"""Run reconstructions LOCALLY on the saved test split — no ./data/ needed.

Reads the test set out of experiments/ (reference_arr.npy = truth,
input_arr.npy = u3232, idx_lst_test.npy = sensors), so it works even though the
raw .npy/.npz are gone. ~26 s per snapshot on Apple GPU (MPS).

TWO TRAPS this script handles — do not "simplify" either away:
  1. SCALER. The pipeline derives mean/std from the TRAIN split, which we no
     longer have. We substitute the test-split stats (vorticity is exactly
     zero-mean by construction) and then VALIDATE by rerunning a stored snapshot
     and comparing to its recorded score. Do not skip that check.
  2. 3-FRAME STACK. slice2sequence saves only the MIDDLE frame, so a sample's
     input must be rebuilt as [saved[i-1], saved[i], saved[i+1]] within one
     trajectory. Feeding one frame three times "works" and is silently 34% wrong.

Written 2026-08-05. See the inpainting-dead-block memory for full context.
"""
import argparse, glob, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from functions.dead_block import parse_size, block_for_trajectory, surviving_sensors
from runners.rs256_guided_diffusion import StdScaler, voriticity_residual
from runners.stochastic_interpolant import StochasticInterpolant

B   = 'experiments/kmflow_re1000_rs256_ddim_conditional_new'
IDX = int(os.environ.get('IDX', 670))          # 0..1271, trajectory-major
BLK = os.environ.get('BLOCK', '16')
NST = int(os.environ.get('STEPS', 100))
PER, S_FULL = 318, 256

def load(folder, pat):
    fs = sorted(glob.glob(os.path.join(B, folder, 'sample_batch*', pat)),
                key=lambda f: int(os.path.basename(os.path.dirname(f)).replace('sample_batch','')))
    return np.concatenate([np.load(f) for f in fs]).astype(np.float32)

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device {dev} | snapshot {IDX} | block {BLK} | {NST} SDE steps\n')

gt_all   = load('si_guided_recons_u3232_t400_r20_w0.0', 'reference_arr.npy')
blur_all = load('guided_recons_u3232_t400_r20_w0.0',    'input_arr.npy')
saved    = load('si_guided_recons_u3232_t400_r20_w0.0', 'sample_arr_run_0_it0.npy')

# Scaler: the pipeline uses TRAIN-split stats, which need ./data/. Substitute the
# test-split stats (vorticity has exactly zero mean by construction, and the two
# splits come from the same generator) -- then VALIDATE by reproducing a known run.
scaler = StdScaler(float(gt_all.mean()), float(gt_all.std()))
print(f'scaler: mean {gt_all.mean():+.6f}  std {gt_all.std():.6f}  (test-split substitute)')

cfg = yaml.safe_load(open('configs/kmflow_re1000_rs256_conditional.yml'))
def d2n(c):
    n = argparse.Namespace()
    for k, v in c.items(): setattr(n, k, d2n(v) if isinstance(v, dict) else v)
    return n
cfg = d2n(cfg); cfg.device = dev
si = StochasticInterpolant(cfg, dev, physics='none', bridge='si')
ck = torch.load('pretrained_weights/si_ckpt.pth', map_location=dev)
si.model.load_state_dict(ck['model']); si.model.eval()

def rmse(a, b): return float(np.sqrt(((a-b)**2).mean()))

def stack3(arr, i):
    """Rebuild a sample's 3-frame stack from the saved MIDDLE frames.

    slice2sequence stores only frame i+1 of each sample, so sample i (frames
    i, i+1, i+2) is [saved[i-1], saved[i], saved[i+1]] within the same
    trajectory. Needed because the network takes 3 channels and the physics
    residual differentiates across them -- feeding one frame 3x is NOT the same
    input and gives a materially different reconstruction (verified).
    """
    t, k = divmod(i, PER)
    if not (1 <= k <= PER - 2):
        raise SystemExit(f'IDX {i} is at a trajectory edge (k={k}); pick 1..{PER-2} within a trajectory')
    lo = t * PER
    return np.stack([arr[lo + k - 1], arr[lo + k], arr[lo + k + 1]])

def run(x0_np, tag):
    x0 = scaler(torch.from_numpy(x0_np[None]).to(dev))
    t0 = time.time()
    with torch.no_grad():
        out = si.sample(x0, n_steps=NST, scaler=scaler)
    if dev.type == 'mps': torch.mps.synchronize()
    y = scaler.inverse(out)[0, 1].cpu().numpy()      # middle frame, as the pipeline saves
    print(f'  {tag:22s} {time.time()-t0:5.1f} s')
    return y

gt, blur = gt_all[IDX], blur_all[IDX]              # middle frames, for scoring
blur3 = stack3(blur_all, IDX)                      # (3,256,256) model input

print('\n[1] VALIDATION — rerun the unmodified u3232 input and compare to the saved run')
rep = run(blur3, 'no block')
print(f'      saved run RMSE vs truth : {rmse(saved[IDX], gt):.4f}')
print(f'      this  run RMSE vs truth : {rmse(rep, gt):.4f}')
d = abs(rmse(rep, gt) - rmse(saved[IDX], gt))
print(f'      difference              : {d:.4f}  '
      + ('OK — scaler substitute is sound' if d < 0.15 else '*** too large, scaler suspect'))

print(f'\n[2] DEAD BLOCK {BLK}')
traj = IDX // PER
bh, bw = parse_size(BLK, S_FULL, S_FULL)
y0, x0c = block_for_trajectory(BLK, 0, traj, S_FULL, S_FULL, None)
idx = np.load('idx_lst_test.npy')[traj]
keep, n_dead = surviving_sensors(idx, (y0, x0c), BLK, S_FULL, S_FULL)
print(f'      {bh}x{bw} at (y={y0}, x={x0c}) = {100*bh*bw/S_FULL**2:.2f}% of the field; '
      f'{n_dead}/{len(idx)} sensors dead')
void3 = blur3.copy(); void3[:, y0:y0+bh, x0c:x0c+bw] = 0.0   # same hole in all 3 frames
void = void3[1]
rec = run(void3, f'void {bh}x{bw}')

m = np.zeros_like(gt, bool); m[y0:y0+bh, x0c:x0c+bw] = True
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a*b).sum() / (np.sqrt((a**2).sum()*(b**2).sum()) + 1e-12))
print(f'\n      inside the hole   RMSE {rmse(rec[m], gt[m]):.3f}   '
      f'null(zeros) {float(np.sqrt((gt[m]**2).mean())):.3f}   '
      f'skill {1-rmse(rec[m],gt[m])/float(np.sqrt((gt[m]**2).mean())):+.1%}   '
      f'corr {corr(rec[m], gt[m]):+.4f}')
print(f'      outside           RMSE {rmse(rec[~m], gt[~m]):.3f}   corr {corr(rec[~m], gt[~m]):+.4f}')
np.save(f'local_snapshot_{BLK.replace("%","pct")}.npy',
        np.stack([gt, void, rec]))
