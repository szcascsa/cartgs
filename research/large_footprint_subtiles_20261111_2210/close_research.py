"""只清理本轮可重建中间二进制/对象；保护基线、最终so、模型和全部指标。"""
from pathlib import Path
import hashlib
import json
import shutil
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    for name in ('FORMAL_CONTROLLER.json','POST_CHECKS.json','ATTRIBUTION_CONTROLLER.json'):
        assert json.loads((HERE/'reports'/name).read_text())['state']=='complete'
    f=json.loads((HERE/'reports/FROZEN.json').read_text())
    final=Path(f['binary']).resolve();assert sha(final)==f['binary_sha256']
    assert all(sha(HERE/n)==h for n,h in f['sources'].items())
    from footprint_study import SCENES
    (HERE/'reports/INPUT_CATALOG.json').write_text(json.dumps(SCENES,indent=2,ensure_ascii=False,default=str))
    origin=HERE.parent/'render_count_scaling_20261111_2022/reports/INPUT_SHA256.json'
    target=HERE/'reports/INPUT_SHA256.json';assert not target.exists();shutil.copy2(origin,target)
    summary=json.loads((HERE/'reports/AGGREGATE/SUMMARY.json').read_text())
    assert not summary['full_gate']['common']
    selection=dict(recommended='previous_F2',replaces_production=False,training_changed=False,
        optional_research_mode=49,optional_mode_status='仅研究保留，未通过全部验收门限',
        reason='同二进制控制消除大部分微小优势；Full逐组双轮门限未通过；不宣称解决大footprint瓶颈。',time=time.time())
    (HERE/'reports/FINAL_SELECTION.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False))
    names=['occupancy_first.so','compact_pixel_first.so','subtiles_first.so','subtiles_diagnostics.so',
        'subtiles_alpha.so','subtiles_macros.so','subtiles_early.so','direct_tiles_failure.so','direct_tiles_fixed.so',
        'short_cache_first.so','cache_policies.so','shared_alpha.so','profile01.sqlite','plot_inputs.tar.gz','attribution_outputs.tar.gz']
    paths=[HERE/'reports'/n for n in names]+list((HERE/'build').glob('*.o'))
    paths=[p for p in paths if p.exists()]
    rows=[]
    for path in paths:
        assert not path.is_symlink() and path.is_file()
        resolved=path.resolve();assert resolved.is_relative_to(HERE.resolve()) and resolved!=final
        rows.append(dict(path=str(resolved),bytes=path.stat().st_size,sha256=sha(path),
            reason='本轮可重建中间产物；对应源码快照/日志保留，Nsight原始rep保留。'))
    wanted={r['path'] for r in rows}
    for process in Path('/proc').glob('[0-9]*/maps'):
        try:text=process.read_text()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(p in text for p in wanted),str(process)
    record=dict(state='planned',time=time.time(),free_before=shutil.disk_usage(HERE).free,
        files=rows,total_bytes=sum(r['bytes'] for r in rows),protected_binary=str(final),protected_sha256=f['binary_sha256'])
    manifest=HERE/'reports/CLEANUP.json';assert not manifest.exists()
    manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    for path in paths:path.unlink()
    assert sha(final)==f['binary_sha256']
    record.update(state='complete',free_after=shutil.disk_usage(HERE).free,ended_at=time.time())
    manifest.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    print('OWN_REBUILDABLE_CLEANUP',len(rows),record['total_bytes'])


if __name__=='__main__':main()
