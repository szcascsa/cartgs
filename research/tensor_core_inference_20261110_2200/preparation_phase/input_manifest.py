"""补齐真实输入文件SHA；只读，不复制大模型。"""
from prep_study import *
from transfer_validation import SLUGS

protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
specs=[s for s in SCENES if s['dataset_kind']=='Replica' or s['slug'] in SLUGS]
rows=[]
for spec in specs:
    root=Path(spec['checkpoint']);paths=[root/'selector.pt',root/'selector_config.yaml']
    clouds=list((root/'point_cloud').glob('iteration_*/point_cloud.ply'));assert len(clouds)==1
    paths+=clouds
    if spec['dataset_kind']=='Replica':paths.append(MODEL_ROOT/spec['slug']/'continuous_calibration/best.pt')
    for path in paths:
        assert path.is_file(),path
        digest=protected.get(str(path)) or sha(path)
        rows.append(dict(scene=spec['id'],path=str(path),bytes=path.stat().st_size,sha256=digest,
            protected_by_original_manifest=str(path) in protected))
save(HERE/'reports/INPUT_CHECKPOINTS.json',dict(files=rows,models_copied=False,datasets_copied=False,
    provenance='初始保护清单内SHA在最终完整性核验中重新检查；其余文件本次直接读取SHA。'))
print('INPUT_FILE_SHA_MANIFEST',len(rows),flush=True)
