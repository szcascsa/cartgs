"""开发集通过后锁定唯一组合；不得按保留集结果修改参数。"""
from pathlib import Path
import argparse,hashlib,json,time
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--selector',type=int,required=True);p.add_argument('--compensator',type=int,required=True);p.add_argument('--warps',type=int,required=True);p.add_argument('--comp-warps',type=int);p.add_argument('--tune',required=True);args=p.parse_args()
assert not (HERE/'reports/FROZEN_CANDIDATE.json').exists()
binary=HERE/'build/mycartgs_tensor_deployment_v1.so'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'source').iterdir() if p.is_file()}
result=dict(frozen_at=time.time(),selector=args.selector,compensator=args.compensator,warps=args.warps,comp_warps=args.comp_warps or args.warps,
    binary=str(binary),binary_sha256=sha(binary),sources=sources,tune=args.tune,
    baseline='/home/prmi/zwj/experiments/compensator_distillation_20261108/build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so',
    baseline_sha256='fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601',
    precision='TC multiply with FP32 accumulation and FP32 physical decoding',training_modified=False,production_deployed=False)
(HERE/'reports/FROZEN_CANDIDATE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('TENSOR_CANDIDATE_FROZEN',result)
