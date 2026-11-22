"""只释放本轮可由源码/Trace重建的中间文件，保留.so和所有关键证据。"""
from pathlib import Path
import hashlib,json,shutil,time

HERE=Path(__file__).resolve().parent;manifest=HERE/'reports/CLEANUP.json';assert not manifest.exists()
paths=[]
for name in ['build','build_v2','build_v3','build_v4']:
    directory=HERE/name
    for p in directory.glob('*.o'):paths.append(p)
for name in ['profile_G01.sqlite','profile_suite01.sqlite']:paths.append(HERE/'evaluation'/name)
removed=[];free_before=shutil.disk_usage(HERE).free
for path in paths:
    path.resolve().relative_to(HERE.resolve());assert not path.is_symlink()
    if not path.is_file():continue
    row=dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.unlink();removed.append(row)
manifest.write_text(json.dumps(dict(checked_at=time.time(),removed=removed,freed_bytes=sum(x['bytes'] for x in removed),
    reason='本轮可重建object及已导出CSV且保留nsys-rep的SQLite；不触及其他目录/模型/原始指标',
    free_before=free_before,free_after=shutil.disk_usage(HERE).free),ensure_ascii=False,indent=2))
print('OWN_REBUILDABLE_FILES_REMOVED',len(removed),sum(x['bytes'] for x in removed))
