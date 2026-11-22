"""保护已有生产/模型/后端，预先锁定测速与质量口径。"""
from common import *
before=json.loads((BASE/'reports/PROTECTED_BEFORE.json').read_text())
for p in BASE.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:before[str(p)]=sha(p)
for p in (FROZEN/'source/python').glob('*.py'):before[str(p)]=sha(p)
for spec in SCENES:
    p=FROZEN/'models'/spec['slug']/'continuous_calibration/best.pt';assert p.is_file();before[str(p)]=sha(p)
assert not (HERE/'reports/PROTECTED_BEFORE.json').exists()
for p,value in before.items():assert sha(p)==value,p
save(HERE/'reports/PROTECTED_BEFORE.json',before)
save(HERE/'reports/PROTOCOL.json',dict(started_at=time.time(),scope='学生专用Head调度、32维专用循环、8点SoA与向量解码；只冻结CPU推理，不训练部署',
    primary_ratios=[.1,.2,.3,.4,.5],scenes=[s['id'] for s in SCENES],conditions=CONDITIONS,
    benchmark='Teacher64/64四个两层Head vs对应Student32/32直接Head，同一Master/Selector/Feature/固定键/相机/选点。CPU单池双轮，端到端4场景双轮',
    stages='准备、Selector、选点、Compensator及输出；分段重复计时单列，不与未插桩总延迟相加',
    quality='分开报告Student相对Teacher蒸馏损失、CPU学生相对普通学生部署误差；不把旧质量门限失败称等价替代',
    backend_quality_gate=dict(mean_psnr_loss_db=.05,per_condition_mean_psnr_loss_db=.2),images_saved=False,
    teacher_default=True,production_training_checkpoints_shared_conda_untouched=True,commit_push_deploy=False))
print('STUDENT_BOOTSTRAP',len(before),flush=True)
