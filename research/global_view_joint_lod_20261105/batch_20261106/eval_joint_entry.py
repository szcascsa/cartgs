"""只增加名义Anchor严格局部ratio入口；复用原全轨迹/双目/指标流程。"""
from pathlib import Path
import hashlib
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
if local:
    sys.path[:0]=[str(R/'evaluation_runtime'),str(R/'source/python')]
    import anchor_frustum
    from anchor_budget import FixedBudgetRenderer

    class LocalRatioRenderer:
        def __init__(self,model,gaussians,ratio,config,temperature,protection):
            self.ratio=ratio
            self.renderer=FixedBudgetRenderer(model,gaussians,
                dict(config,temperature=temperature,protection_enabled=protection),seed=0)
        def prepare(self,camera):
            data,ids,stats=self.renderer.prepare(camera,ratio_budget=self.ratio)
            stats['legacy_quantile_budget']=False
            return data,ids,stats
    anchor_frustum.AnchorFrustumRenderer=LocalRatioRenderer
    sys.argv.append('--selector-anchor-frustum')

# 只对本批内存中的脚本文本去掉巨量逐帧ID保存；保持选点、渲染、指标不变。
# 单帧身份改为SHA256，既可核验也不会保存数GB的冗余ID数组。
script=R/'source/python/run.py'
text=script.read_text()
old="selected_rows.append(selected_ids.cpu().numpy().astype(np.int32,copy=False))"
assert text.count(old)==1
text=text.replace(old,"selected_rows.append(__import__('hashlib').sha256(selected_ids.cpu().numpy().tobytes()).hexdigest())")
start=text.index('        offsets=np.concatenate(([0],np.cumsum(')
end=text.index('        aggregate_keys =',start)
text=text[:start]+"        with open(os.path.join(output_result_path,'selection_hashes.json'),'w') as stream:\n            json.dump(selected_rows,stream)\n"+text[end:]
# 用runpy调用方提供的路径执行，替换临时脚本只有本批使用。
patched=R/'source/python/run_metrics_only.py'
if not patched.exists() or patched.read_text()!=text:patched.write_text(text)
sys.argv[1]=str(patched)
sys.argv[0]=str(R/'lod_eval_entry.py')
runpy.run_path(sys.argv[0],run_name='__main__')
if local:
    output=Path(sys.argv[sys.argv.index('--output_path')+1])
    stats=list(csv.DictReader((output/'anchor_frustum_per_view.csv').open()))
    summary=json.loads((output/'metric_summary.json').read_text())
    summary.update(evaluation_policy='visible_nominal_anchor_ratio_strict_K',
        target_K_mean=sum(float(r['requested_budget']) for r in stats)/len(stats),
        candidate_N_mean=sum(float(r['candidate_gaussians']) for r in stats)/len(stats),
        budget_shortfall_views=sum(int(r['budget_shortfall'])>0 for r in stats))
    summary['target_K']=summary['target_K_mean']
    (output/'metric_summary.json').write_text(json.dumps(summary,indent=2))
    path=output/'selector_stats.json'
    metadata=json.loads(path.read_text())
    metadata.update(selection_policy='visible_ratio_floor_rN_stable_exact_K',
        target_active=summary['target_K_mean'],budget_population=summary['candidate_N_mean'],
        actual_vs_target=summary['actual_K_mean']/max(1.,summary['target_K_mean']))
    path.write_text(json.dumps(metadata,indent=2))
