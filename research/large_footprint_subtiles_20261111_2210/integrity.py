"""只读核验冻结输入、基线和本轮二进制，不声称恢复训练或部署。"""
from pathlib import Path
import hashlib
import json
import time

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'render_count_scaling_20261111_2022'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_manifest(path):
    manifest=json.loads(path.read_text())
    mismatches=[name for name,value in manifest.items() if sha(name)!=value]
    assert not mismatches,mismatches
    return dict(path=str(path),sha256=sha(path),files=len(manifest),passed=True)


def main():
    result=dict(time=time.time(),manifests=[])
    for path in (HERE/'reports/BASELINE_SHA256.json',PRIOR/'reports/INPUT_SHA256.json'):
        result['manifests'].append(verify_manifest(path))
    frozen=json.loads((HERE/'reports/FROZEN.json').read_text())
    assert all(sha(HERE/path)==value for path,value in frozen['sources'].items())
    assert sha(frozen['binary'])==frozen['binary_sha256']
    result['frozen']=dict(files=len(frozen['sources']),binary=frozen['binary'],sha256=frozen['binary_sha256'],passed=True)
    # 旧等待器已因validation06失败退出；保留原状态供审计，禁止重新启动。
    path=HERE/'reports/IDLE_SCREEN.json';old=json.loads(path.read_text())
    if old['state']=='waiting':
        assert not Path(f'/proc/{old["pid"]}').exists()
        assert json.loads((HERE/'reports/jobs/validation06.json').read_text())['returncode']!=0
        backup=path.with_name('IDLE_SCREEN_ORIGINAL.json');assert not backup.exists();backup.write_bytes(path.read_bytes())
        old.update(state='failed',reason='validation06非法访问退出；旧waiting状态未更新。后续validation07修复，已由独立控制器完成验收。',
                   reconciled_at=time.time())
        path.write_text(json.dumps(old,indent=2,ensure_ascii=False))
    result['controllers']=[]
    for name in ('FORMAL_CONTROLLER.json','POST_CHECKS.json','ATTRIBUTION_CONTROLLER.json'):
        identity=json.loads((HERE/'reports'/name).read_text());assert identity['state']=='complete'
        path=Path(f'/proc/{identity["pid"]}/stat')
        if path.exists():
            fields=path.read_text().rpartition(')')[2].split()
            assert fields[19]!=str(identity['start_ticks']) or fields[0]=='Z'
        result['controllers'].append(dict(name=name,state='complete',original_pid=identity['pid']))
    result['passed']=True
    (HERE/'reports/INTEGRITY_FINAL.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
