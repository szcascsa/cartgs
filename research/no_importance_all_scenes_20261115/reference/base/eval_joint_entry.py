"""全轨迹数值评价；局部显式使用真实视角，全局保持零条件。"""
from pathlib import Path
import csv
import json
import runpy
import sys
import torch

R=Path(__file__).resolve().parent
local='--joint-local-ratio' in sys.argv
if local:sys.argv.remove('--joint-local-ratio')
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path[:0]=[str(R/'source/python'),str(R/'evaluation_runtime')]
if local:
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRatioRenderer:
        def __init__(self,model,gaussians,ratio,config,temperature,protection):
            assert int(model.joint_view_conditioning_schema)==2
            self.ratio=ratio
            self.renderer=FixedBudgetRenderer(model,gaussians,dict(config,temperature=temperature,protection_enabled=protection),seed=0,backend='pytorch')
        def prepare(self,camera):
            data,ids,stats=self.renderer.prepare(camera,ratio_budget=self.ratio,view_conditioned=True)
            assert stats['view_conditioned'] and len(ids)==min(stats['requested_budget'],stats['eligible_candidates'])
            stats['legacy_quantile_budget']=False
            return data,ids,stats
    anchor_frustum.AnchorFrustumRenderer=LocalRatioRenderer
    sys.argv.append('--selector-anchor-frustum')
sys.argv[1]=str(R/'source/python/run_metrics_only.py')
sys.argv[0]=str(R/'lod_eval_entry.py')
runpy.run_path(sys.argv[0],run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output_path')+1]);summary=json.loads((output/'metric_summary.json').read_text())
summary.update(view_conditioned=local,conditioning='both_real_view' if local else 'zero_view_zero_mode',li_weight=.1)
if local:
    rows=list(csv.DictReader((output/'anchor_frustum_per_view.csv').open()))
    summary.update(evaluation_policy='visible_nominal_anchor_ratio_strict_K',target_K_mean=sum(float(v['requested_budget']) for v in rows)/len(rows),
        candidate_N_mean=sum(float(v['candidate_gaussians']) for v in rows)/len(rows),budget_shortfall_views=sum(int(v['budget_shortfall'])>0 for v in rows))
    summary['target_K']=summary['target_K_mean']
    file=output/'selector_stats.json';metadata=json.loads(file.read_text())
    metadata.update(selection_policy='visible_ratio_floor_rN_stable_exact_K',target_active=summary['target_K_mean'],
        budget_population=summary['candidate_N_mean'],view_conditioned=True,
        actual_vs_target=summary['actual_K_mean']/max(1.,summary['target_K_mean']))
    file.write_text(json.dumps(metadata,indent=2))
(output/'metric_summary.json').write_text(json.dumps(summary,indent=2))
