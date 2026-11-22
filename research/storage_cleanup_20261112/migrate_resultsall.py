"""迁移指定旧Vanilla，保留必要数据适配，清理用户授权的其余历史实验。"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess

BASE=Path('/home/prmi/zwj')
ROOT=BASE/'mycartgs/resultsall'
VANILLA='vanilla_all_scenes_resumable_20261025'
ADAPTER='r3_cuda_mono_vector_anchor0_twopass_20261024_111150'
DELETE_NAMES={
    'spatial_fourier_rgbd_fullres_full_lod_20261027','scale_chain_fix_20261028',
    'batch_20261018_tum_aligned','r3_cuda_rgbd_twopass_20261023_210938',
    'batch_20261016','flexgs_failed_scene_retest_20261028','spatial_fourier_rgbd_full_lod_20261027',
    '.vanilla_last4_setup_20261017','batch_20261016_nolog','flexgs_root_fix_research_20261028',
    'vanilla_tail_comparison_20261026',ADAPTER,'batch_20261017_vanilla_last4',
    'r3_cuda_tail_schedule_2x2_high_ratios_20261025','r3_cuda_rgbd_anchor_only1_after_20261023_210938',
    'spatial_fourier_remaining_fullres_full_lod_20261027','vanilla_summary_20261017_153227',
    'r3_cuda_tail_rotation_anchor0_aligned_20261024_151706','r3_cuda_tum_rgbd_high_ratios_20261025',
    'fourier_failure_root_cause_20261028','batch_20261017_aligned','.aligned_setup_20261017',
}


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    result={}
    for folder,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            path=Path(folder,name);relative=str(path.relative_to(root))
            if path.is_symlink():result[relative]=dict(symlink=os.readlink(path))
            elif path.is_file():result[relative]=dict(size=path.stat().st_size,sha256=sha(path))
    return result


def usage(path):
    return int(subprocess.check_output(['du','-sx','-B1',str(path)],text=True).split()[0])


def active_references():
    hits=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=os.readlink(proc/'cwd')
            mapped=(proc/'maps').read_text(errors='replace')
            files=[]
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if str(ROOT) in target:files.append(target)
                except OSError:pass
            if str(ROOT) in cmd or str(ROOT) in cwd or str(ROOT) in mapped or files:
                hits.append(dict(pid=int(proc.name),command=cmd[:1000],cwd=cwd,open_files=files))
        except (OSError,PermissionError):pass
    return hits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True);parser.parse_args()
    assert ROOT.resolve()==ROOT and ROOT.is_dir() and not ROOT.is_symlink()
    source=ROOT/VANILLA;destination=BASE/'realLOD/vanilla'/VANILLA
    adapter_source=ROOT/ADAPTER/'data'
    adapter_destination=BASE/'realLOD/data_adapters/resultsall_legacy_20261024'/ADAPTER/'data'
    assert {p.name for p in ROOT.iterdir()}==DELETE_NAMES|{VANILLA}
    assert source.is_dir() and not source.is_symlink() and not destination.exists()
    assert adapter_source.is_dir() and not adapter_source.is_symlink() and not adapter_destination.exists()
    assert source.stat().st_dev==destination.parent.stat().st_dev
    assert not active_references(),'存在仍访问待清理目录的进程，停止删除'
    timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    audit=BASE/'realLOD/reports'/('resultsall_cleanup_'+timestamp);audit.mkdir(parents=True)
    free_before=shutil.disk_usage(BASE).free
    old_usage={p.name:usage(p) for p in ROOT.iterdir()}
    vanilla_before=inventory(source);adapter_before=inventory(adapter_source)
    write(audit/'VANILLA_FILE_HASHES_BEFORE.json',vanilla_before)
    write(audit/'ADAPTER_FILE_HASHES_BEFORE.json',adapter_before)
    protected={}
    for project in [BASE/'mycartgs',BASE/'realLOD']:
        for folder in ['src','include','cfg','bin','lib']:
            root=project/folder
            if not root.is_dir() or root.is_symlink():continue
            for directory,dirs,files in os.walk(root,followlinks=False):
                dirs[:]=[n for n in dirs if not Path(directory,n).is_symlink()]
                for name in files:
                    path=Path(directory,name)
                    if path.is_file() and not path.is_symlink():protected[str(path)]=sha(path)
    for filename in ['LOD_PIPELINE.json','reports/LOD_METRICS.csv']:
        f=BASE/'realLOD/ablation'/filename
        if f.is_file():protected[str(f)]=sha(f)
    write(audit/'PROTECTED_FILES_BEFORE.json',protected)
    write(audit/'PLAN.json',dict(authorization='用户指定保留旧Vanilla并删除resultsall其他历史实验',
        source=str(source),destination=str(destination),adapter_source=str(adapter_source),
        adapter_destination=str(adapter_destination),delete_directories=sorted(DELETE_NAMES),
        measured_bytes=old_usage,free_before=free_before,active_references=[]))
    print('PREPARATION_VERIFIED',str(audit),'vanilla_files',len(vanilla_before),flush=True)
    os.rename(source,destination)
    source.symlink_to(destination,target_is_directory=True)
    adapter_destination.parent.mkdir(parents=True)
    os.rename(adapter_source,adapter_destination)
    adapter_source.symlink_to(adapter_destination,target_is_directory=True)
    assert inventory(destination)==vanilla_before
    assert inventory(adapter_destination)==adapter_before
    summary=json.loads((destination/'summary.json').read_text())
    assert len(summary)==99 and all(row['state']=='complete' for row in summary)
    current_index=[]
    for row in summary:
        checkpoint=Path(row['checkpoint'])
        checkpoint.relative_to(source)
        assert checkpoint.is_dir(),str(checkpoint)
        relocated=destination/checkpoint.relative_to(source)
        assert relocated.is_dir()
        current_index.append(dict(id=row['id'],historical_checkpoint=str(checkpoint),current_checkpoint=str(relocated)))
    db=sqlite3.connect('file:'+str(destination/'queue.sqlite')+'?mode=ro',uri=True)
    assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok';db.close()
    (destination/'HISTORICAL_RECORD.md').write_text(
        '# 2026年10月25日旧Vanilla历史实验记录\n\n'
        '本目录是之前跑的原版CaRtGS Vanilla实验，共99次成功训练，seed为0/1/2。'
        '它与2026年10月31日后来独立训练、seed0重复三遍的新Vanilla批次不同。'
        '此目录作为历史记录归档，不作为当前生产默认模型。\n\n'
        '2026年11月12日从以下位置完整迁入，模型、点云、指标、原日志和数据库保持逐文件SHA-256一致：\n\n'
        +str(source)+'\n\n当前实际存储位置：\n\n'+str(destination)+'\n\n'
        '旧路径仅保留兼容软链接，该批次checkpoint的原绝对路径仍可读取。'
        '其他已清理历史批次的来源路径仅作为原始记录保留，不保证原队列可直接重启。'
        'CURRENT_CHECKPOINT_INDEX.json提供99个checkpoint的当前路径；ARCHIVE_METADATA.json记录迁移身份。'
        '迁移未启动训练或评价。历史队列不应自动重启。\n\n迁移审计：'+str(audit)+'\n')
    write(destination/'CURRENT_CHECKPOINT_INDEX.json',current_index)
    write(destination/'ARCHIVE_METADATA.json',dict(historical=True,label='之前跑的旧Vanilla记录',
        experiment_date='2026-10-25',archived_at=datetime.datetime.now().isoformat(),seeds=[0,1,2],
        successful_runs=99,source=str(source),destination=str(destination),compatibility_symlink=str(source),
        audit=str(audit),original_files_sha256_verified=len(vanilla_before),production_default=False))
    print('MIGRATION_HASHES_AND_99_CHECKPOINTS_VERIFIED',flush=True)
    assert not active_references(),'迁移后出现新的旧目录访问，停止后续删除'
    deleted=[]
    for name in sorted(DELETE_NAMES):
        target=ROOT/name
        assert target.parent==ROOT and target.resolve().parent==ROOT and not target.is_symlink()
        assert target.name!=VANILLA
        if name==ADAPTER:
            for child in list(target.iterdir()):
                if child.name=='data':
                    assert child.is_symlink() and child.resolve()==adapter_destination
                    continue
                assert child.parent==target
                if child.is_symlink():child.unlink()
                elif child.is_dir():shutil.rmtree(child)
                else:child.unlink()
            assert {p.name for p in target.iterdir()}=={'data'}
        else:shutil.rmtree(target)
        deleted.append(name)
        write(audit/'DELETE_PROGRESS.json',dict(completed=deleted))
    assert {p.name for p in ROOT.iterdir()}=={VANILLA,ADAPTER}
    assert source.is_symlink() and source.resolve()==destination
    assert adapter_source.is_symlink() and adapter_source.resolve()==adapter_destination
    for relative,record in vanilla_before.items():
        f=destination/relative
        assert dict(symlink=os.readlink(f))==record if f.is_symlink() else dict(size=f.stat().st_size,sha256=sha(f))==record
    assert inventory(adapter_destination)==adapter_before
    assert all(Path(f).is_file() and sha(Path(f))==digest for f,digest in protected.items())
    free_after=shutil.disk_usage(BASE).free
    receipt=dict(passed=True,completed_at=datetime.datetime.now().isoformat(),vanilla_destination=str(destination),
        successful_runs=99,vanilla_original_files_verified=len(vanilla_before),adapter_destination=str(adapter_destination),
        adapter_original_files_verified=len(adapter_before),protected_files_verified=len(protected),
        historical_directories_cleaned=len(deleted),estimated_removed_allocated_bytes=sum(old_usage[n] for n in DELETE_NAMES)-usage(adapter_destination),
        free_before_bytes=free_before,free_after_bytes=free_after,observed_free_increase_bytes=free_after-free_before,
        resultsall_remaining_bytes=usage(ROOT),compatibility_links=[str(source),str(adapter_source)],audit=str(audit))
    write(audit/'RESULT.json',receipt)
    (audit/'清理结果.md').write_text('# resultsall迁移与清理核验\n\n'
        '旧Vanilla已完整迁移并标注历史身份，99个checkpoint路径和SQLite完整性核验通过。'
        '必要VECtor/TUM适配数据迁入realLOD/data_adapters，旧路径仅保留软链接。'
        '其余22个历史实验目录已清理（数据适配原父目录仅留兼容链接）。'
        '原Vanilla文件、适配文件、生产源码配置及二进制哈希验证通过；未运行训练或评价。\n\n'
        '实测分区可用空间：%.2f GiB → %.2f GiB；resultsall本身仅剩%d字节。\n'
        %(free_before/2**30,free_after/2**30,receipt['resultsall_remaining_bytes']))
    print('CLEANUP_COMPLETE',json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
