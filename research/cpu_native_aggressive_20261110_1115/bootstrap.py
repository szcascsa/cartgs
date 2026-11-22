"""建立本轮独立协议和只读保护清单，不复制数据/模型。"""
from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parent
root=Path('/home/prmi/zwj/experiments')
before=json.loads((root/'large_footprint_20261110_0912/reports/PROTECTED_BEFORE.json').read_text())
before.update(json.loads((root/'cpu_mlp_benchmark_20261110_1045/reports/cpu_formal01/PROTECTED_BEFORE.json').read_text()))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()
assert not (HERE/'reports/PROTECTED_BEFORE.json').exists()
for name,expected in before.items():assert sha(name)==expected,name
(HERE/'reports/PROTECTED_BEFORE.json').write_text(json.dumps(before,indent=2))
scenes=json.loads((root/'sparse_rasterizer_sixhour_20261110_0053/reports/SCENES.json').read_text())
protocol=dict(started_at=time.time(),authorization='速度优先，允许部署FP32微差；全部准备在CPU，GPU只消费属性渲染',
    tune=['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd'],
    validation=[s['slug'] for s in scenes if s['slug'] not in ['Replica__office2__rgbd','Replica__room0__rgbd','TUM_RGBD__freiburg1_xyz__rgbd']],
    constraints='保持网络、权重、FP32、预算和同分规则；不训练、不改生产/原模型/共享环境、不commit/push',
    quality_screen=dict(mean_gt_psnr_loss_db=.05,per_condition_mean_loss_db=.2,
        policy='研究筛查门槛；报告每帧/每条件实际PSNR/SSIM/LPIPS、ID重合及属性误差，拒绝明显质量下降'),
    optimizations=['C++原生调度','AVX2 FMA','首层静态分解','每Anchor投影复用','固定6线程池','精确分区选择','有界pinned输出'],
    final_conditions=['local0.01','local0.05','local0.1','local0.3','local0.7','local0.03','local0.08','local0.2','local0.4','local0.6','fixed0','fixed1000','fixed5000','global0.1','full'],
    baseline='本机CPU PyTorch相同真实权重/固定噪声；另与已有GPU部署入口渲染质量配对',
    no_same_frame_cache=True,max_projected_cache_bytes=1024**3,full_training_untouched=True)
(HERE/'reports/PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
(HERE/'reports/SCENES.json').write_text(json.dumps(scenes,indent=2));print('CPU_BOOTSTRAP_COMPLETE',len(before))
