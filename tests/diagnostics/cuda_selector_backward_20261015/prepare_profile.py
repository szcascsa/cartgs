"""冻结本轮基线，生成只用于 NVTX 分解的源码副本。"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

D = Path(__file__).resolve().parent
root = D.parents[2]
frozen = D / 'frozen'
frozen.mkdir(exist_ok=True)
files = ['src/selector_cuda.cu', 'src/selector_cuda.cpp', 'include/selector_cuda.h']
candidate = '--candidate' in sys.argv
if not candidate:
    manifest = {}
    for name in files:
        target = frozen / Path(name).name
        if not target.exists():target.write_bytes((root/name).read_bytes())
        data = target.read_bytes()
        manifest[name] = hashlib.sha256(data).hexdigest()
    manifest_path=D/'baseline_sources.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==manifest, '冻结源码 hash 已改变'
    else:manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
s = (root/'src/selector_cuda.cu' if candidate else frozen/'selector_cuda.cu').read_text()
s = '#include <nvToolsExt.h>\n#include <string>\n' + s
s = s.replace('namespace {', '''namespace {
struct Range { Range(const std::string& s){nvtxRangePushA(s.c_str());} ~Range(){nvtxRangePop();} };
''', 1)
s = s.replace('  auto handle=at::cuda::getCurrentCUDABlasHandle();',
    '  Range range("gemm_m"+std::to_string(m)+"_n"+std::to_string(n)+"_k"+std::to_string(k)+"_"+ta+tb);\n  auto handle=at::cuda::getCurrentCUDABlasHandle();')
s = s.replace('  // Strict candidate keeps', '  Range range("bias_rows"+std::to_string(dy.size(0))+"_cols"+std::to_string(dy.size(1)));\n  // Strict candidate keeps')
s = s.replace('  int rows=x.size(0),in=x.size(1),out=dy.size(1);',
    '  Range range("linear_vjp_in"+std::to_string(x.size(1))+"_out"+std::to_string(dy.size(1)));\n  int rows=x.size(0),in=x.size(1),out=dy.size(1);')
s = s.replace('  int n=x[0].size(0);SpatialArgs args{};',
    '  Range range("combined_spatial_relu_input_vjp");\n  int n=x[0].size(0);SpatialArgs args{};')
s = s.replace('    if(grouped && b<3)continue;', '    Range range("branch_"+std::to_string(b));\n    if(grouped && b<3)continue;')
(D / ('profile_candidate.cu' if candidate else 'profile_selector.cu')).write_text(s)
