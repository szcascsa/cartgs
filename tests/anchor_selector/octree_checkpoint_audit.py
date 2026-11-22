"""checkpoint 只读验证：层级、occupancy、native parity、forward 与显存。"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/"python"))
from selector import load_selector,load_selector_metadata
from gaussian_model import GaussianModel
root,name=Path(sys.argv[1]),sys.argv[2]
checkpoint=root/name/"10000_shutdown/ply";cloud=checkpoint/"point_cloud/iteration_10000"
model=load_selector(str(checkpoint/"selector.pt"),"cuda")
g=GaussianModel(3);g.load_ply(str(cloud/"point_cloud.ply"))
metadata=load_selector_metadata(str(cloud/"selector_metadata.ply"),"cuda")
probe=torch.jit.load(str(root/"validation"/(name+"_probe.pt")),map_location="cuda").state_dict()
bank=model.anchors;n=len(g.get_xyz);m=len(bank.features)
levels=bank.levels.cpu().numpy() if hasattr(bank,"levels") else np.zeros(m,dtype=np.int64)
branches=bank.branches.cpu().numpy() if hasattr(bank,"branches") else np.zeros(m,dtype=bool)
counts=torch.bincount(bank.rows,minlength=m).cpu().numpy()
assert len(bank.rows)==n and bank.rows.min()>=0 and bank.rows.max()<m and counts.sum()==n
if getattr(model,"explicit_blocks",False):
    assert n==10*m and (counts==10).all() and len(torch.unique(bank.block_uids))==m
assert levels.min()>=0 and levels.max()<=2 and not counts[branches].any()
if hasattr(bank,"levels"):assert not ((counts>128)&(levels<2)).any()
assert all(torch.isfinite(t).all() for t in model.state_dict().values())
assert (metadata["maturity"]==1).all()
def forward(r):
    return model(g.get_xyz,None,g.get_scaling,torch.tensor([[r]],device="cuda"),sh_dc=g.get_features[:,:1],opacity=g.get_opacity)
with torch.no_grad():
    parity=[]
    for i,r in enumerate((.01,.05,.1,.15)):
        p=forward(r)
        error=float((p-probe["probability_"+str(i)]).abs().max())
        assert error<1e-6 and torch.equal(metadata["cached_uniform"]<p,probe["mask_"+str(i)])
        parity.append(dict(ratio=r,max_probability_error=error,mask_exact=True))
    timings=[]
    for repeat in range(5):
        for _ in range(20):forward(.1)
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(200):forward(.1)
        end.record();end.synchronize()
        timings.append(start.elapsed_time(end)/200)
    temporary_bytes=torch.cuda.max_memory_allocated()-base
keys=bank.keys.cpu().numpy();centers=bank.centers.cpu().numpy()
cell_centers=(keys+.5)*(float(bank.voxel_size)/2.**levels)[:,None]
if getattr(model,"global_xyz",False):
    assert np.allclose(centers,cell_centers,rtol=1e-6,atol=1e-7)
    assert np.allclose(bank.offset_scales.cpu().numpy().ravel(),float(bank.voxel_size)/2.**levels)
with (root/"validation"/(name+"_cells.csv")).open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["node_id","level","is_branch","occupancy","key_x","key_y","key_z","cell_center_x","cell_center_y","cell_center_z","offset_ref_x","offset_ref_y","offset_ref_z","h","over_capacity_l2"])
    for j in range(m):writer.writerow([j,int(levels[j]),bool(branches[j]),int(counts[j]),*keys[j].tolist(),*cell_centers[j].tolist(),*centers[j].tolist(),float(bank.voxel_size)/2**int(levels[j]),bool(levels[j]==2 and counts[j]>128)])
np.savez_compressed(root/"validation"/(name+"_mapping.npz"),rows=bank.rows.cpu().numpy(),keys=keys,levels=levels,branches=branches,occupancy=counts,cell_centers=cell_centers,offset_reference=centers)
if getattr(model,"explicit_blocks",False):
    np.savez_compressed(root/"validation"/(name+"_block_identity.npz"),uids=bank.block_uids.cpu().numpy(),
        keys=keys,levels=levels,reference_centers=centers,reference_scales=bank.offset_scales.cpu().numpy())
summary=dict(name=name,population=n,feature_count=m,feature_bytes=bank.features.numel()*bank.features.element_size(),
    global_xyz=getattr(model,"global_xyz",False),input_width=model.local_emd[0].weight.shape[1],
    feature_plus_two_adam_moments_bytes=bank.features.numel()*bank.features.element_size()*3,
    layout_buffer_bytes=sum(t.numel()*t.element_size() for t in bank.buffers()),
    node_count_by_level={str(l):int((levels==l).sum()) for l in range(3)},
    occupied_leaf_count_by_level={str(l):int(((levels==l)&(~branches)&(counts>0)).sum()) for l in range(3)},
    retained_empty_nodes=int((counts==0).sum()),occupied_leaf_count=int((counts>0).sum()),
    occupancy_quantiles=np.quantile(counts[counts>0],[0,.5,.9,.99,1]).tolist(),
    l2_over_capacity_count=int(((levels==2)&(counts>128)).sum()),
    l2_over_capacity_gaussians=int(counts[(levels==2)&(counts>128)].sum()),
    forward_ms_repeats=timings,forward_ms_median=float(np.median(timings)),forward_temporary_peak_bytes=temporary_bytes,
    native_forward_ms_repeats=probe["native_forward_ms"].cpu().tolist(),native_forward_ms_median=float(probe["native_forward_ms"].median()),
    native_forward_measurement="C++ LibTorch forward; precomputed activated attributes, 20 warmups, 5x200 calls, synchronized wall time; actual N reported",
    forward_measurement="Python-equivalent LibTorch CUDA forward; 20 warmups, 5x200 iterations, CUDA events; includes activated attribute access; not end-to-end Mapping latency",
    checkpoint_parity=parity)
if "refinement_continuity_error" in probe:
    summary["real_map_continuity_probability_error"]=float(probe["refinement_continuity_error"])
    summary["real_map_continuity_splits"]=int(probe["refinement_continuity_splits"])
if "refinement_jump_error" in probe:
    summary["real_map_split_max_probability_change"]=float(probe["refinement_jump_error"])
    summary["real_map_split_mean_probability_change_by_ratio"]=probe["refinement_jump_mean_probability_error"].cpu().tolist()
    summary["real_map_split_changed_cached_mask_fraction_by_ratio"]=probe["refinement_jump_changed_mask_fraction"].cpu().tolist()
    summary["real_map_splits"]=int(probe["refinement_continuity_splits"])
(root/"validation"/(name+"_checkpoint_audit.json")).write_text(json.dumps(summary,indent=2))
print("checkpoint audit",name,summary["occupied_leaf_count_by_level"],summary["forward_ms_median"],flush=True)
