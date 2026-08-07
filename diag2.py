import numpy as np, yaml
from functions.dead_block import voronoi_fill, read_npz_trajectories
cfg=yaml.safe_load(open("configs/kmflow_re1000_rs256_conditional.yml"))["data"]
gt=np.asarray(np.load(cfg["data_dir"],mmap_mode="r")[-4:,:2],dtype=np.float32)
with np.load(cfg["sample_data_dir"],allow_pickle=True) as f: idx=f["idx_lst"][-4:].astype(np.int64)
u=read_npz_trajectories(cfg["sample_data_dir"],cfg["data_kw"],slice(-4,None),n_frames=2)
H=W=256; t=0; fr=0
iy,ix=np.divmod(idx[t],W)
sensor_vals=np.unique(gt[t,fr][iy,ix])          # exact float32 values, no rounding
flat=u[t,fr].ravel()
inset=np.isin(flat, sensor_vals)
print(f"u3232 pixels whose value is EXACTLY some sensor value: {100*inset.mean():.2f}%")
print(f"   (100% => nearest-neighbour fill;  ~1.6% => only the sensors themselves)")
mine=voronoi_fill(gt[t,fr],idx[t],H,W)
d=np.abs(mine-u[t,fr]); bad=d>1e-6
print(f"my fill differs at {100*bad.mean():.2f}% of pixels")
print(f"   of those, u3232's value is a sensor value: {100*np.isin(u[t,fr][bad],sensor_vals).mean():.2f}%")
# is u3232 smooth between sensors? compare local variation
from scipy.ndimage import distance_transform_edt
mask=np.zeros(H*W,bool); mask[idx[t]]=True; mask=mask.reshape(H,W)
lap_u=np.abs(np.diff(u[t,fr],axis=0)).mean(); lap_m=np.abs(np.diff(mine,axis=0)).mean()
lap_g=np.abs(np.diff(gt[t,fr],axis=0)).mean()
print(f"mean |d/dy|:  gt {lap_g:.4f}   u3232 {lap_u:.4f}   my NN fill {lap_m:.4f}")
print("   (NN fill is blocky => LARGER gradient than a smooth interpolant)")
