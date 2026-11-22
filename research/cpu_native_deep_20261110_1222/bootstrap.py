"""冻结本轮协议、上一版及原模型保护清单；无数据集或模型复制。"""
from common import *
before=json.loads((OLD/'reports/PROTECTED_BEFORE.json').read_text())
for path in OLD.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts and 'local_before_canonical_sync' not in path.parts:before[str(path)]=sha(path)
assert not (HERE/'reports/PROTECTED_BEFORE.json').exists()
for path,value in before.items():assert sha(path)==value,path
save(HERE/'reports/PROTECTED_BEFORE.json',before)
protocol=json.loads((OLD/'reports/PROTOCOL.json').read_text())
protocol.update(started_at=time.time(),baseline=str(OLD),work='仅C++调度/交接＋AVX2微核；冻结Teacher推理',
    acceptance='相同身份/预算；优先V4逐位属性一致；否则GPU质量平均损失<=.05dB/条件平均<=.2dB并单列最差帧；只保留可重复收益',
    independent_variants=['control','schedule','kernel','combined'],not_in_scope=['训练','模型变小','量化','渲染器改动','新选点算法'],
    final_rounds=2,thread_environment=dict(OMP_WAIT_POLICY='PASSIVE',KMP_BLOCKTIME='0',GOMP_SPINCOUNT='0'))
save(HERE/'reports/PROTOCOL.json',protocol);print('DEEP_BOOTSTRAP',len(before),flush=True)
