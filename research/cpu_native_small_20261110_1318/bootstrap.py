"""冻结协议及生产/原模型/两轮历史证据哈希保护。"""
from common import *
before=json.loads((BASE/'reports/PROTECTED_BEFORE.json').read_text())
for p in BASE.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:before[str(p)]=sha(p)
assert not (HERE/'reports/PROTECTED_BEFORE.json').exists()
for name,value in before.items():assert sha(name)==value,name
save(HERE/'reports/PROTECTED_BEFORE.json',before)
protocol=json.loads((BASE/'reports/PROTOCOL.json').read_text())
protocol.update(started_at=time.time(),baseline=str(BASE),scope='小输出层/顺序选点压缩/线程调度长尾；仅冻结CPU推理',
    defaults='研究目录独立开关，候选冻结前只用三个开发场景，21组保留验证',
    constraints='训练、选点、Master、原checkpoint、原渲染器不改，不commit/push/部署',
    quality='优先相对上一版逐位一致；若出现误差须独立GPU质量验收；平均损失<=.05dB、条件平均<=.2dB并报告最差帧',
    measurement='6物理核/6原生线程、Torch辅助1线程；每次只驻留一个线程池；相同运动/跳转相机、双轮顺序轮换')
save(HERE/'reports/PROTOCOL.json',protocol);print('SMALL_BOOTSTRAP',len(before),flush=True)
