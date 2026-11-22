"""从真实Mapping trace核对在线成本及checkpoint能否由生产Python读取。"""
import argparse,json,sys
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--name',default='online_spatial');a=parser.parse_args()
sys.path.insert(0,'/home/prmi/zwj/mycartgs/python')
from selector import load_selector
from selector import compensated_attributes
from gaussian_model import GaussianModel
completion=json.loads((ROOT/a.name/'completion.json').read_text());cp=Path(completion['checkpoint'][0])
model=load_selector(str(cp/'selector.pt'),'cuda');assert hasattr(model,'spatial_compensation')
assert not hasattr(model,'gaussian_compensation') and not hasattr(model.anchors,'compensation_ratio_knots')
assert bool(model.spatial_compensation.normalization_ready)
cloud=next((cp/'point_cloud').glob('iteration_*'));gaussians=GaussianModel(3);gaussians.load_ply(str(cloud/'point_cloud.ply'))
with torch.no_grad():
    probability=model(gaussians.get_xyz,gaussians.get_rotation,gaussians.get_scaling,
        torch.tensor([[.075]],device='cuda'),sh_dc=gaussians.get_features_dc,opacity=gaussians.get_opacity)
    attributes=compensated_attributes(model,gaussians,(probability>.5).float(),.075)
    assert attributes is not None and all(torch.isfinite(v).all() for v in attributes.values())
trace=pd.read_csv(ROOT/a.name/'mapping_trace.csv');post=trace[trace.compensation>0]
cols=['total','selector_forward','compensation','lod_render','lod_loss','lod_backward','optimizer','full_render','full_backward','operations','densify']
stats=dict(all_mapping_steps=len(trace),actual_compensation_steps=len(post),first_compensation_step=int(post.iteration.min()),post_compensation_host_ms=post[cols].mean().to_dict(),all_mapping_host_ms=trace[cols].mean().to_dict(),normalization_ready=True,private_tables=False,compensation_parameters=sum(p.numel() for p in model.spatial_compensation.parameters()),N=len(model.anchors.rows),anchors=len(model.anchors.features),checkpoint=str(cp),python_loaded_attributes_finite=True)
(ROOT/(a.name+'_inspection.json')).write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
