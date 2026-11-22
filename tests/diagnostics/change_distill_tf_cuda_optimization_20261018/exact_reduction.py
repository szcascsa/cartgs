"""按已安装 LibTorch Reduce.cuh 的小维度归约树匹配原 quaternion。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase2_exact'
assert not out.exists()
shutil.copytree(D/'phase2/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill_transport.cu'
s=p.read_text().replace('a*w-((b*u+c*v)+d*z)', 'a*w-((b*u+d*z)+c*v)')
s=s.replace('((o0*o0+o1*o1)+o2*o2)+o3*o3', '(o0*o0+o1*o1)+(o2*o2+o3*o3)')
p.write_text(s)
