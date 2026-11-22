"""仅在本轮独立目录建立研究现场，冻结F2/输入/既有TC身份与墙钟预算。"""
from pathlib import Path
from datetime import datetime,timedelta
import hashlib
import json
import shutil
import time

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
F2=EXP/'render_count_scaling_20261111_2022/background_stage'
OLD=EXP/'large_footprint_subtiles_20261111_2210'
TC=EXP/'tensor_core_inference_20261110_2200'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not (HERE/'reports/PROTOCOL.json').exists()
    assert shutil.disk_usage(HERE).free>=5*(1<<30)
    for folder in ('reports','logs','build','figures'):(HERE/folder).mkdir(exist_ok=True)
    start=datetime.fromisoformat('2026-11-12T00:11:00+08:00')
    protocol=dict(start=start.isoformat(),deadline=(start+timedelta(hours=9)).isoformat(),
        render_review=(start+timedelta(hours=4)).isoformat(),render_freeze_latest=(start+timedelta(hours=6)).isoformat(),
        final_validation_reserve=(start+timedelta(hours=8)).isoformat(),phase='rendering',
        quality=dict(equal_condition_mean_psnr_loss_max=.20,scene_ratio_mean_psnr_loss_max=.50),
        meaningful_render_latency_reduction=.10,tune=['Replica__office2__rgbd','Replica__room0__rgbd',
            'TUM_RGBD__freiburg1_xyz__rgbd','VECtor__corner-slow__stereo'],
        protected=['production','training','checkpoints','datasets','shared_conda'],no_commit_push_deploy=True,
        workload_A='固定视角和属性，按冻结随机排列嵌套删点',workload_B='原网络的真实LoD补偿输出',
        note='原00:11保守起点不重置；编译、等待、试验、分析和整理均计入预算。')
    (HERE/'reports/PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    shutil.copytree(F2/'source',HERE/'source')
    for name in ('launch.py','job_runner.py'):shutil.copy2(OLD/name,HERE/name)
    manifests={}
    for directory in (F2/'source',TC/'preparation_phase/source'):
        for path in directory.rglob('*'):
            if path.is_file():manifests[str(path)]=sha(path)
    for path in [F2/'build/mycartgs_render_background_scaling_v1.so',
        TC/'reports/FROZEN_CANDIDATE.json',TC/'preparation_phase/reports/FROZEN_JOINT_CANDIDATE.json']:
        manifests[str(path)]=sha(path)
    (HERE/'reports/PROTECTED_SHA256.json').write_text(json.dumps(manifests,indent=2))
    shutil.copy2(EXP/'render_count_scaling_20261111_2022/reports/INPUT_SHA256.json',HERE/'reports/INPUT_SHA256.json')
    (HERE/'reports/STATUS.json').write_text(json.dumps(dict(state='running',phase='analysis',time=time.time())))
    print('NINE_HOUR_SITE_CREATED',protocol['start'],protocol['deadline'],len(manifests))


if __name__=='__main__':main()
