"""仅清理本轮可重建对象和SQLite；长诊断无损压缩并保留删除清单。"""
from pathlib import Path
import gzip,hashlib,json,time

HERE=Path(__file__).resolve().parent
assert HERE.name=='sparse_rasterizer_sixhour_20261110_0053'
manifest_path=HERE/'reports/CLEANUP_MANIFEST.json'
previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
rows=[]
allowed=[HERE/'build',HERE/'build_graph',HERE/'late_candidates/microtile/build',HERE/'evaluation']
for root in allowed:
    if not root.exists():continue
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in ['.o','.sqlite']:
            target=path.resolve();assert target.is_relative_to(HERE) and not path.is_symlink()
            rows.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),action='delete_rebuildable_object_or_sqlite'))
            path.unlink()
for root in [HERE/'logs',HERE/'reports']:
    for path in sorted(root.glob('*')):
        if not path.is_file() or path.suffix not in ['.log','.txt'] or path.stat().st_size<1<<20:continue
        target=path.resolve();assert target.is_relative_to(HERE) and not path.is_symlink()
        raw=path.read_bytes();compressed=path.with_suffix(path.suffix+'.gz')
        if compressed.exists():assert gzip.decompress(compressed.read_bytes())==raw
        else:
            with gzip.open(compressed,'xb',compresslevel=9) as stream:stream.write(raw)
        assert gzip.decompress(compressed.read_bytes())==raw
        rows.append(dict(path=str(path.relative_to(HERE)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            action='lossless_gzip_then_remove_original',retained=str(compressed.relative_to(HERE)),compressed_bytes=compressed.stat().st_size))
        path.unlink()
new_count=len(rows);rows=previous.get('files',[])+rows
result=dict(time=time.time(),first_cleanup_time=previous.get('first_cleanup_time',previous.get('time',time.time())),
    root=str(HERE),files=rows,removed_bytes=sum(r['bytes'] for r in rows),
    reclaimed_bytes=sum(r['bytes']-r.get('compressed_bytes',0) for r in rows),
    preserved='原基线和上一版所有文件；最终源码/二进制；原模型/数据；全部计时CSV与失败摘要；生产/共享缓存不触碰')
manifest_path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
print('CLEANUP_COMPLETE',new_count,'new;',len(rows),'total;',result['reclaimed_bytes'])
