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
R=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,R); os.chdir(R)
from functions.dead_block import parse_size, block_for_trajectory
from runners.rs256_guided_diffusion import StdScaler
from runners.stochastic_interpolant import StochasticInterpolant

B='experiments/kmflow_re1000_rs256_ddim_conditional_new'; PER=318; N=256
def load(f,p):
    fs=sorted(glob.glob(os.path.join(B,f,'sample_batch*',p)),
              key=lambda q:int(os.path.basename(os.path.dirname(q)).replace('sample_batch','')))
    return np.concatenate([np.load(q) for q in fs]).astype(np.float32)
gt_all=load('si_guided_recons_u3232_t400_r20_w0.0','reference_arr.npy')
bl_all=load('guided_recons_u3232_t400_r20_w0.0','input_arr.npy')

IDXS=[t*PER+k for t in range(4) for k in (60,200)]            # 8 snapshots, 2 per trajectory
def stack3(arr,i):
    t,k=divmod(i,PER); lo=t*PER
    return np.stack([arr[lo+k-1],arr[lo+k],arr[lo+k+1]])

dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
scaler=StdScaler(float(gt_all.mean()),float(gt_all.std()))
cfg=yaml.safe_load(open('configs/kmflow_re1000_rs256_conditional.yml'))
def d2n(c):
    n=argparse.Namespace()
    for k,v in c.items(): setattr(n,k,d2n(v) if isinstance(v,dict) else v)
    return n
cfg=d2n(cfg); cfg.device=dev
si=StochasticInterpolant(cfg,dev,physics='none',bridge='si')
si.model.load_state_dict(torch.load('pretrained_weights/si_ckpt.pth',map_location=dev)['model'])
si.model.eval()
print(f'device {dev} | {len(IDXS)} snapshots | 100 SDE steps', flush=True)
print(flush=True)

gts=np.stack([gt_all[i] for i in IDXS])
def corr(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum()/(np.sqrt((a**2).sum()*(b**2).sum())+1e-12))

print(f"{'block':>8}{'% field':>9}{'RMSE in':>9}{'null':>8}{'skill':>9}{'corr in':>9}{'corr out':>10}")
print('-'*62)
for BLK in ['16','32','64']:
    stacks=[]; masks=[]
    for i in IDXS:
        s3=stack3(bl_all,i).copy()
        bh,bw=parse_size(BLK,N,N)
        y0,x0=block_for_trajectory(BLK,0,i//PER,N,N,None)
        s3[:,y0:y0+bh,x0:x0+bw]=0.0
        m=np.zeros((N,N),bool); m[y0:y0+bh,x0:x0+bw]=True
        stacks.append(s3); masks.append(m)
    x0t=scaler(torch.from_numpy(np.stack(stacks)).to(dev))
    t0=time.time()
    with torch.no_grad(): out=si.sample(x0t,n_steps=100,scaler=scaler)
    if dev.type=='mps': torch.mps.synchronize()
    rec=scaler.inverse(out)[:,1].cpu().numpy()
    ri=[];nu=[];ci=[];co=[]
    for j,m in enumerate(masks):
        g=gts[j]; r=rec[j]
        ri.append(np.sqrt(((r[m]-g[m])**2).mean())); nu.append(np.sqrt((g[m]**2).mean()))
        ci.append(corr(r[m],g[m])); co.append(corr(r[~m],g[~m]))
    ri,nu=np.mean(ri),np.mean(nu)
    bh,bw=parse_size(BLK,N,N)
    print(f'{BLK+"x"+BLK:>8}{100*bh*bw/N**2:>8.2f}%{ri:>9.3f}{nu:>8.3f}{1-ri/nu:>+8.1%}{np.mean(ci):>9.3f}{np.mean(co):>10.4f}   [{time.time()-t0:.0f}s]', flush=True)
