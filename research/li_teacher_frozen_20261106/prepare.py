"""建立独立冻结诊断目录，记录模型和生产文件保护清单。"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/view_budget_raw18_20261105')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/li_teacher_frozen_20261106')
    assert json.loads((OLD/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    for n in ['build','runtime','runtime/lib','evaluation','evaluation/reports','configs','runs']:(ROOT/n).mkdir(exist_ok=True)
    assert not (ROOT/'reports/PRODUCTION_BEFORE.json').exists()
    protected=json.loads((OLD/'reports/PRODUCTION_BEFORE.json').read_text())
    before={n:dict(bytes=Path(n).stat().st_size,sha256=sha(Path(n))) for n in protected}
    (ROOT/'reports/PRODUCTION_BEFORE.json').write_text(json.dumps(before,indent=2))
    checkpoint=OLD/'runs/full_raw_li01/23560_shutdown/ply'
    inputs={str(p):sha(p) for p in checkpoint.rglob('*') if p.is_file()}
    inputs[str(checkpoint.parent.parent/'CameraTrajectory_TUM.txt')]=sha(checkpoint.parent.parent/'CameraTrajectory_TUM.txt')
    inputs[str(OLD/'evaluation/reports/PER_VIEW.csv')]=sha(OLD/'evaluation/reports/PER_VIEW.csv')
    original=OLD/'runtime/lib/libcuda_rasterizer.so';library=ROOT/'runtime/lib/libcuda_rasterizer.so'
    shutil.copy2(original,library);assert sha(library)==sha(original)
    (ROOT/'reports/INPUTS_BEFORE.json').write_text(json.dumps(inputs,indent=2))
    (ROOT/'configs/experiment.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),dataset='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
        ratios=[.01,.05],seed=0,frames=792,bank=72,primary_teacher_source='uncompensated Full alpha*T/HW; native Training Forward',
        selectors=['base','view','teacher'],compensation=[False,True],gaussians=13298,anchors=2790,
        old_metrics=str(OLD/'evaluation/reports/PER_VIEW.csv'),min_mean_gain_dB=.10,learning_or_adam_steps=0),indent=2))
    (ROOT/'reports/LIBRARY_IDENTITY.json').write_text(json.dumps(dict(original=str(original),copied=str(library),sha256=sha(library)),indent=2))
    shutil.copyfile(Path(__file__).with_name('PROTOCOL.md'),ROOT/'reports/PROTOCOL.md')
    print(json.dumps(dict(protected_files=len(before),input_files=len(inputs),free_GiB=shutil.disk_usage(ROOT).free/1024**3)),flush=True)


if __name__=='__main__':main()
