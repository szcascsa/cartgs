"""用户授权：results中只保留vanilla，删除其余16个顶层条目。"""
from pathlib import Path
import argparse
import datetime
import json
import os
import shutil
import sys

sys.path.insert(0,'/home/prmi/zwj/realLOD')
from migrate_resultsall_20261112 import inventory,sha,usage,write

ROOT=Path('/home/prmi/zwj/mycartgs/results')
KEEP=ROOT/'vanilla'
DELETE_NAMES={
    'tf_capacity_initial_evidence_20261012.tar.gz','change_distill_ablation','interactive_renders',
    'tf_capacity_phase1_numeric_20261012.tar.gz','change_distill_performance_audit',
    'fr1_room_r3_20261022_184714','tf_capacity_phase1_evidence_20261012.tar.gz',
    'tf_capacity_ratio_adapter_20261012.tar.gz','tf_capacity_diagnostics_evidence_20261012.tar.gz',
    'change_distill_split_epochs','cudaselector','change_distill_optimization','cudatfnewprotection',
    'change_distill_tf_cuda_optimization','interactive','change_distill_tf_multi_ratio',
}


def active_references():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(proc/'cwd');mapped=(proc/'maps').read_text(errors='replace')
            opened=[]
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if str(ROOT)+'/' in target:opened.append(target)
                except OSError:pass
            if str(ROOT)+'/' in cmd or cwd==str(ROOT) or cwd.startswith(str(ROOT)+'/') or str(ROOT)+'/' in mapped or opened:
                found.append(dict(pid=int(proc.name),command=cmd[:1000],cwd=cwd,open_files=opened))
        except (OSError,PermissionError):pass
    return found


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True);parser.parse_args()
    assert ROOT.resolve()==ROOT and ROOT.is_dir() and not ROOT.is_symlink()
    assert KEEP.is_dir() and not KEEP.is_symlink()
    assert {p.name for p in ROOT.iterdir()}==DELETE_NAMES|{'vanilla'}
    assert not active_references(),'存在访问待清理目录的进程，停止删除'
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    audit=Path('/home/prmi/zwj/realLOD/reports')/('results_cleanup_'+stamp);audit.mkdir()
    before=inventory(KEEP)
    for name,record in before.items():
        if 'symlink' not in record:continue
        target=(KEEP/name).resolve()
        assert ROOT not in target.parents or KEEP in target.parents,'保留目录有指向待删除内容的依赖'
    write(audit/'VANILLA_FILE_HASHES_BEFORE.json',before)
    prior=Path('/home/prmi/zwj/realLOD/reports/resultsall_cleanup_20261112_210034/PROTECTED_FILES_BEFORE.json')
    protected={name:sha(Path(name)) for name in json.loads(prior.read_text())}
    write(audit/'PROTECTED_FILES_BEFORE.json',protected)
    sizes={name:usage(ROOT/name) for name in sorted(DELETE_NAMES)}
    free_before=shutil.disk_usage(ROOT).free;keep_before=usage(KEEP)
    write(audit/'PLAN.json',dict(authorization='用户明确指定results仅保留vanilla，其余全部删除',
        root=str(ROOT),keep=str(KEEP),delete_items=sizes,keep_allocated_bytes=keep_before,
        free_before_bytes=free_before,active_references=[]))
    print('PREPARATION_VERIFIED',str(audit),'keep_files',len(before),'delete_items',len(sizes),flush=True)
    assert not active_references()
    deleted=[]
    for name in sorted(DELETE_NAMES):
        target=ROOT/name
        assert target.parent==ROOT and target!=KEEP and target.name!='vanilla'
        if target.is_symlink():target.unlink()
        else:
            assert target.resolve().parent==ROOT
            if target.is_dir():shutil.rmtree(target)
            else:target.unlink()
        deleted.append(name);write(audit/'DELETE_PROGRESS.json',dict(completed=deleted))
    assert {p.name for p in ROOT.iterdir()}=={'vanilla'}
    assert inventory(KEEP)==before
    assert all(Path(name).is_file() and sha(Path(name))==digest for name,digest in protected.items())
    free_after=shutil.disk_usage(ROOT).free
    receipt=dict(passed=True,completed_at=datetime.datetime.now().isoformat(),root=str(ROOT),
        retained_directory=str(KEEP),retained_files_verified=len(before),protected_files_verified=len(protected),
        deleted_items=deleted,removed_allocated_bytes=sum(sizes.values()),retained_allocated_bytes=usage(KEEP),
        free_before_bytes=free_before,free_after_bytes=free_after,observed_free_increase_bytes=free_after-free_before,
        audit=str(audit))
    write(audit/'RESULT.json',receipt)
    (audit/'清理结果.md').write_text('# results仅保留vanilla清理核验\n\n'
        '根据用户授权，删除results中除vanilla之外的全部16个顶层条目（11个目录、5个压缩文件）。'
        '保留目录原文件逐项SHA-256一致，生产源码、配置、二进制及近期消融状态和指标哈希一致。\n\n'
        '保留目录：'+str(KEEP)+'\n\n实测分区可用空间：%.2f GiB → %.2f GiB。\n'
        %(free_before/2**30,free_after/2**30))
    print('CLEANUP_COMPLETE',json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
