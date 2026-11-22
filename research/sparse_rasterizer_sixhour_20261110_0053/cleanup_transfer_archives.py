"""交付已逐文件校验后，仅删除本轮临时传输包与被替换的首版打包文件。"""
from pathlib import Path
import hashlib,json,time

HERE=Path(__file__).resolve().parent
assert HERE.name=='sparse_rasterizer_sixhour_20261110_0053'
verified=json.loads((HERE/'LOCAL_DELIVERY_VERIFIED.json').read_text())
receipt=json.loads((HERE/'DELIVERY_RECEIPT.json').read_text())
final=HERE/receipt['archive']
assert verified['passed'] and verified['archive_sha256']==receipt['sha256']
assert hashlib.sha256(final.read_bytes()).hexdigest()==receipt['sha256']
names=['ANALYSIS_EVIDENCE.tar.gz','PROFILE_EVIDENCE.tar.gz','FINAL_PLOT_INPUTS.tar.gz',
    'FINAL_ASYNC_PLOT_INPUTS.tar.gz','FINAL_FIGURES.tar.gz','FINAL_ACCEPTED_FIGURES.tar.gz',
    'FINAL_HELPERS.tar.gz','DELIVERY_HELPERS.tar.gz','FINAL_REVISION_HELPERS.tar.gz',
    'FINAL_ACCEPTANCE_HELPERS.tar.gz','FINAL_ACCEPTED_HELPERS.tar.gz',
    'LOW_POINT_RENDERING_SIX_HOUR_DELIVERY.previous.tar.gz']
rows=[]
for name in names:
    path=HERE/name
    if not path.exists():continue
    assert path.is_file() and not path.is_symlink() and path.resolve().parent==HERE and path!=final
    rows.append(dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reason='本轮临时传输归档；最终交付包和已解包文件已全部SHA验证'))
    path.unlink()
result=dict(passed=True,time=time.time(),files=rows,reclaimed_bytes=sum(row['bytes'] for row in rows),
    preserved_archive=receipt,original_baselines_models_and_all_metrics_untouched=True,
    scope='独立研究目录内精确文件白名单，不递归删除、不使用共享缓存')
(HERE/'reports/TRANSFER_CLEANUP.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('TRANSFER_CLEANUP_COMPLETE',len(rows),result['reclaimed_bytes'])
