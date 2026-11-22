"""导出真实运行产物及已保存特征统计；不启动测试或重新训练。"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import torch

root=Path('/home/prmi/zwj/anchor-shared16-ratio8-20261026')
repo=Path('/home/prmi/zwj/mycartgs')
out=root/'validation'
runtime=json.loads((out/'training_runtime.json').read_text())
rows=[]
architectures=[]
for name,item in runtime.items():
    module=torch.jit.load(str(Path(item['checkpoint'])/'selector.pt'),map_location='cpu')
    state=module.state_dict()
    n=len(state['anchors.rows']);m=len(state['anchors.features'])
    architectures.append(dict(model=name,population=n,anchors=m,selector_input=state['local_emd.0.weight'].shape[1],
        scalar_mode=int(state.get('anchor_ratio_scalar_input',0)),
        scalar_column_norm=float(state['local_emd.0.weight'][:,-1].norm()) if 'anchor_ratio_scalar_input' in state else 0.,
        compensation_input=state['compensation.0.weight'].shape[1],shared_feature_width=state['anchors.features'].shape[1],
        ratio_feature_tables=sum(k.startswith('anchors.ratio_features_') for k in state),
        compensation_feature_tables=sum(k.startswith('anchors.compensation_features_') for k in state)))
    for key,value in state.items():
        if key=='anchors.features' or key.startswith(('anchors.ratio_features_','anchors.compensation_features_','gaussian_compensation.residual_')):
            value=value.detach().double()
            rows.append(dict(model=name,table=key,rows=value.shape[0],width=value.shape[1],parameters=value.numel(),
                finite=bool(torch.isfinite(value).all()),mean=float(value.mean()),std=float(value.std(unbiased=False)),
                rms=float(value.square().mean().sqrt()),max_abs=float(value.abs().max()),
                changed_row_fraction=float((value.abs().sum(1)>0).double().mean())))
for filename,data in [('feature_statistics.csv',rows),('model_architecture.csv',architectures)]:
    with (out/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
manifest=json.loads((root/'source_manifest.json').read_text())
(root/'source_after.json').write_text(json.dumps(dict(source_matches_local_manifest=all(
    hashlib.sha256((repo/n).read_bytes()).hexdigest()==h for n,h in manifest.items()),tests_run=False,isolated_build=False)))
with tarfile.open(root/'evidence.tar.gz','w:gz') as tar:
    for p in list(out.glob('*'))+list((root/'preview').glob('*.png'))+[root/'sync_receipt.json',root/'source_after.json']:
        if p.is_file():tar.add(p,arcname=str(p.relative_to(root)))
    for name,item in runtime.items():
        cp=Path(item['checkpoint']);cloud=next((cp/'point_cloud').glob('iteration_*'))
        for p in [cp/'selector.pt',cp/'selector_config.yaml',cp/'cameras.json',cloud/'anchor_selector_state.json',
                  root/name/'input_selector.yaml',root/name/'training.log']:
            tar.add(p,arcname='models/'+name+'/'+p.name)
print('EXPORT_COMPLETE',len(rows),'tables',(root/'evidence.tar.gz').stat().st_size,flush=True)
