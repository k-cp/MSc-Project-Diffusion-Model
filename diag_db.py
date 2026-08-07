import numpy as np, yaml, os
from functions.dead_block import voronoi_fill, read_npz_trajectories
cfg=yaml.safe_load(open("configs/kmflow_re1000_rs256_conditional.yml"))["data"]
N=5
gt=np.asarray(np.load(cfg["data_dir"],mmap_mode="r")[-4:,:N],dtype=np.float32)
with np.load(cfg["sample_data_dir"],allow_pickle=True) as f:
    idx=f["idx_lst"][-4:].astype(np.int64)
u=read_npz_trajectories(cfg["sample_data_dir"],cfg["data_kw"],slice(-4,None),n_frames=N)
H=W=256
t=0
iy,ix=np.divmod(idx[t],W)
print("1) does u3232 equal gt AT the sensors?")
d=np.abs(u[t][:,iy,ix]-gt[t][:,iy,ix])
print(f"   max|u-gt| at sensors = {d.max():.3e}   (0 => sensor set + indexing are correct)")
print("2) my fill vs shipped u3232")
mine=voronoi_fill(gt[t],idx[t],H,W)
diff=np.abs(mine-u[t])
frac=(diff>1e-6).mean()
print(f"   pixels differing: {100*frac:.2f}%   max {diff.max():.3f}   mean {diff.mean():.4f}")
print("3) are the differing pixels EQUIDISTANT ties?")
from scipy.ndimage import distance_transform_edt
mask=np.zeros(H*W,bool); mask[idx[t]]=True; mask=mask.reshape(H,W)
dist=distance_transform_edt(~mask)
bad=diff[0]>1e-6
print(f"   mean distance-to-nearest-sensor: differing px {dist[bad].mean():.2f} vs matching {dist[~bad].mean():.2f}")
print("4) is u3232 a nearest-neighbour fill at all? (its value must equal SOME sensor's value)")
vals=set(np.round(gt[t][0][iy,ix],5))
hit=np.isin(np.round(u[t][0],5),list(vals)).mean()
print(f"   fraction of u3232 pixels whose value is one of the 1024 sensor values: {100*hit:.1f}%")
