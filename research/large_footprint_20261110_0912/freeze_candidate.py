"""调优集结束后锁定候选身份，之后不得按保留场景调参。"""
from pathlib import Path
import argparse,hashlib,json,tarfile,time
from footprint_runtime import MODES

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser=argparse.ArgumentParser();parser.add_argument('--mode',type=int,required=True)
parser.add_argument('--validation',required=True);parser.add_argument('--screen',required=True)
parser.add_argument('--secondary',type=int);args=parser.parse_args()
path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
valid=json.loads((HERE/'reports'/args.validation/'SUMMARY.json').read_text())
assert valid['passed'] and MODES[args.mode] in valid['valid_modes']
if args.secondary is not None:assert MODES[args.secondary] in valid['valid_modes']
screen=json.loads((HERE/'reports'/args.screen/'SUMMARY.json').read_text())
job=json.loads((HERE/'reports/jobs'/(args.screen+'.json')).read_text())
assert screen['passed'] and job['measurement_valid'] and job['foreign_active_samples']==0
binary=HERE/'build_v4/large_footprint_graph_v4.so'
assert sha(binary)==valid['binary_sha256']==screen['binary_sha256']
names=[p.relative_to(HERE).as_posix() for p in (HERE/'source').rglob('*') if p.is_file()]
names+=['footprint_runtime.py','study_common.py','paired_benchmark.py','lifecycle.py','profile_candidate.py']
result=dict(frozen=True,tuning_complete=True,mode=args.mode,name=MODES[args.mode],created_at=time.time(),
    validation=args.validation,screen=args.screen,binary=str(binary),binary_sha256=sha(binary),
    additional_modes={} if args.secondary is None else {'D':args.secondary},
    sources={n:sha(HERE/n) for n in sorted(names)},
    reference='A原入口为数值oracle；G/R上一轮最终版为本轮新增收益对照',
    full_policy='允许Full变慢，逐场景报告，不以K/Ratio分流',heldout_opened_after_freeze=True)
path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
with tarfile.open(HERE/'source_frozen.tar.gz','w:gz') as archive:archive.add(HERE/'source',arcname='source')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False,indent=2))
