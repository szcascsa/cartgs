"""从本批冻结快照建立LI=0.5独立对照，不更改原配置和模型。"""
from pathlib import Path
import copy
import hashlib
import json
import re
import shutil

B = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
R = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert R.name == 'dual_view_li05_office34_20261107'
    assert not (R/'LOD_PIPELINE.json').exists()
    source = (B/'configs/dual_view_li.yaml').read_text()
    text, count = re.subn(r'^ViewBudget\.li_weight: 0\.1$', 'ViewBudget.li_weight: 0.5', source, flags=re.M)
    assert count == 1 and text.replace('ViewBudget.li_weight: 0.5', 'ViewBudget.li_weight: 0.1') == source
    (R/'configs/dual_view_li.yaml').write_text(text)
    helpers = ['run_batch.py', 'pipeline.py', 'oom_retry.py', 'lod_eval_entry.py', 'verify_initialization.py', 'test_queue.py']
    for name in helpers:
        shutil.copy2(B/name, R/name)
        assert sha(B/name) == sha(R/name)
    entry = (B/'eval_joint_entry.py').read_text()
    assert entry.count('li_weight=.1') == 1
    (R/'eval_joint_entry.py').write_text(entry.replace('li_weight=.1', 'li_weight=.5'))
    shutil.copytree(B/'runtime', R/'runtime', symlinks=True)
    for name in ['source', 'evaluation_runtime']:
        (R/name).symlink_to(B/name, target_is_directory=True)
    old = json.loads((B/'LOD_PIPELINE.json').read_text())
    keys = ['id', 'dataset', 'scene', 'mode', 'winner', 'full_psnr', 'full_job',
            'full_evaluation', 'camera_config', 'mapping_config', 'candidates', 'target_steps']
    jobs = []
    protected = {str(B/'configs/dual_view_li.yaml'): sha(B/'configs/dual_view_li.yaml')}
    for scene in ['office3', 'office4']:
        job = next(j for j in old['jobs'] if j['id'] == 'Replica/'+scene+'/rgbd')
        fresh = {k: copy.deepcopy(job[k]) for k in keys}
        fresh.update(state='pending', attempt_history=[], evaluations=[], health_failures=0, oom_failures=0, li_weight=.5)
        jobs.append(fresh)
        for folder in [Path(job['full_job']['result']['checkpoint']), Path(job['trained_model'])]:
            for path in folder.rglob('*'):
                if path.is_file():
                    protected[str(path)] = sha(path)
    runtime = {}
    for path in (R/'runtime').rglob('*'):
        if path.is_file():
            relative = path.relative_to(R)
            assert sha(path) == sha(B/relative)
            runtime[str(relative)] = sha(path)
            protected[str(B/relative)] = sha(B/relative)
    state = dict(phase='prepared', jobs=jobs, protocol=copy.deepcopy(old['protocol']),
                 original_batch=str(B), li_weight=.5, changed_configuration=['ViewBudget.li_weight'],
                 baseline_evaluation_must_finish_first=True)
    (R/'LOD_PIPELINE.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    (R/'PROTECTED_BEFORE.json').write_text(json.dumps(protected, indent=2)+'\n')
    receipt = dict(passed=True, changed_lines=['ViewBudget.li_weight: 0.1 -> 0.5'],
                   runtime_sha256=runtime, source_snapshot=str(B/'source'),
                   source_manifest_sha256=sha(B/'reports/SOURCE_HASHES.json'),
                   protected_files=len(protected), steps={j['id']: j['target_steps'] for j in jobs})
    (R/'reports/PREPARED.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    (R/'reports/PROTOCOL.md').write_text(
        '# office3／office4 RGB-D：LI权重0.5额外对照\n\n'
        '先补完原office4 RGB-D的11项评价。原26组队列继续保持HOLD；两组对照结束并汇报后再恢复。\n\n'
        '两组从各自原先选中的Full checkpoint重新开始，seed0、fresh Adam/RNG、固定拓扑、Master可更新。'
        'Full＋全局LoD＋局部LoD，同相机同ratio联合反传后一次Adam；双网络真实视角条件。'
        '只将LI权重0.1改为0.5；训练步数office3为20,249、office4为22,233。\n\n'
        '训练ratio：1%、5%、10%、30%、50%、70%；未见ratio：3%、8%、20%、40%、60%，只评价。'
        '每组评价全局11档、局部11档和Full，共23项完整轨迹。全局按整图预算，局部按名义Anchor视锥候选预算，'
        '同ratio不代表同点数。输出PSNR、SSIM、LPIPS、ATE、N、实际K及逐帧数值，不保存渲染图片。\n\n'
        '沿用原健康检查和OOM证据分类，各最多额外重试2次。失败明确记录，不能按PSNR挑重跑。'
        '与已存在LI=0.1结果比较，不重训基线、不改生产源码或默认配置、不提交或推送。\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
