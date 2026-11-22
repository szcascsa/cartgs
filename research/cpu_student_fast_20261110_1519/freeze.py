"""只在开发集与图像预验通过后冻结最终Student候选。"""
from common import *
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);args=p.parse_args();path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
quality=json.loads((HERE/'reports/quality_tune01/SUMMARY.json').read_text());assert quality['passed'] and quality['variant']==args.variant
names=['source/cpu_student_fast.cpp','source/student_fast.h','source/native_handoff.h','kernel_runtime.py','student_fast_runtime.py','common.py','formal.py']
pool=lambda s:s[s.index('class Pool {'):s.index('// 权重以8个输出')]
assert pool((HERE/'source/cpu_student_fast.cpp').read_text())==pool((KERNEL/'source/cpu_mid.cpp').read_text())
save(path,dict(frozen_at=time.time(),variant=args.variant,binary=extension().__file__,binary_sha256=sha(extension().__file__),sources={n:sha(HERE/n) for n in names},
    baseline=str(BASE),baseline_binary=str(KERNEL/'build/mycartgs_cpu_mid.so'),baseline_binary_sha256=sha(KERNEL/'build/mycartgs_cpu_mid.so'),
    primary_ratios=[.1,.2,.3,.4,.5],development=['Replica/office2/rgbd','Replica/room0/rgbd','Replica/office1/mono'],heldout_scenes=13,
    numerical_contract='Score和UID不变；记录每属性/原始残差差异；优化学生相对原学生平均GT PSNR损失<=.05dB、各条件均值<=.2dB，并报告最差帧',
    pool_unchanged=True,production_deployed=False,training_modified=False,commit_push=False))
print('STUDENT_FAST_FROZEN',args.variant,sha(extension().__file__),flush=True)
