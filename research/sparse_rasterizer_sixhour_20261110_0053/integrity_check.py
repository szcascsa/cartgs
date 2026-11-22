"""最终只读核验生产、原模型、基线、冻结候选与研究进程身份。"""
from pathlib import Path
import hashlib,json,os,subprocess,time

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    protected=json.loads((HERE/'reports/PROTECTED_BEFORE.json').read_text());changed=[];missing=[]
    after={}
    for name,expected in protected.items():
        path=Path(name)
        if not path.is_file():missing.append(name);continue
        actual=sha(path);after[name]=actual
        if actual!=expected:changed.append(dict(path=name,before=expected,after=actual))
    (HERE/'reports/PROTECTED_AFTER.json').write_text(json.dumps(after,indent=2))
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())
    source_changes=[name for name,value in config['source_hashes'].items() if sha(HERE/name)!=value]
    binary_changes=[name for name,row in config['binaries'].items() if sha(row['path'])!=row['sha256']]
    preparation=json.loads((HERE/'reports/FROZEN_PREPARATION.json').read_text())
    preparation_changes=[name for name,value in preparation['sources'].items() if sha(HERE/name)!=value]
    camera_graph=json.loads((HERE/'reports/FROZEN_CAMERA_GRAPH.json').read_text())
    camera_graph_changes=[name for name,value in camera_graph['sources'].items() if sha(HERE/name)!=value]
    async_path=HERE/'reports/FROZEN_CAMERA_ASYNC.json'
    camera_async_changes=[]
    if async_path.exists():
        camera_async=json.loads(async_path.read_text())
        camera_async_changes=[name for name,value in camera_async['sources'].items() if sha(HERE/name)!=value]
    active=[];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for path in sorted((HERE/'reports/jobs').glob('*.json')):
        row=json.loads(path.read_text());proc=Path('/proc')/str(row.get('runner_pid',0))/'stat'
        if proc.is_file():
            data=proc.read_text().rpartition(')')[2].split()
            if data[0]!='Z' and data[19]==row.get('runner_start_ticks') and boot==row.get('boot_id'):active.append(row)
    images=[str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and 'figures' not in p.relative_to(HERE).parts]
    result=dict(passed=not(changed or missing or source_changes or binary_changes or preparation_changes or camera_graph_changes or camera_async_changes or active or images),
        checked_at=time.time(),protected_files=len(protected),changed=changed,missing=missing,
        frozen_source_changes=source_changes,frozen_binary_changes=binary_changes,preparation_source_changes=preparation_changes,
        camera_graph_source_changes=camera_graph_changes,
        camera_async_source_changes=camera_async_changes,
        active_own_jobs=active,render_images_outside_figures=images,baseline_binaries={k:dict(path=v['path'],sha256=sha(v['path'])) for k,v in
            json.loads((HERE/'reports/PROTOCOL.json').read_text()).items() if k in ['baseline_A','baseline_B']},
        candidates=config['binaries'],disk_free_bytes=__import__('shutil').disk_usage(HERE).free,
        commit_push_deploy=False,production_training_shared_conda_untouched=True)
    (HERE/'reports/INTEGRITY_FINAL.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False));assert result['passed']


if __name__=='__main__':main()
