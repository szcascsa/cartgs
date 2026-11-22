"""仅清理本轮两个可重建Sanitizer对象，先核对路径、归属并记录哈希。"""
from pathlib import Path
import hashlib,json,os,time

HERE=Path(__file__).resolve().parent
assert HERE==Path('/home/prmi/zwj/experiments/cpu_student_fast_20261110_1519')
audit=HERE/'reports/CLEANUP_AUDIT.json'
assert not audit.exists()
removed=[]
for name in ['build_asan/cpu_sanitized.o','build_ubsan/cpu_sanitized.o']:
    path=HERE/name
    assert path.is_file() and not path.is_symlink() and path.resolve()==path
    stat=path.stat();assert stat.st_uid==os.getuid()
    removed.append(dict(path=str(path),bytes=stat.st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reason='可由保留源码与build.ninja重建，已保留通过验收的so和日志'))
audit.write_text(json.dumps(dict(time=time.time(),removed=removed,total_bytes=sum(x['bytes'] for x in removed)),ensure_ascii=False,indent=2),encoding='utf-8')
for item in removed:Path(item['path']).unlink()
print('SANITIZER_OBJECTS_CLEANED',sum(x['bytes'] for x in removed))
