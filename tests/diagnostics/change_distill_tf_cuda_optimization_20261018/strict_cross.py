"""匹配 ATen cross 的显式 FMA；nvcc 仍使用 --fmad=false。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase3_strict'
assert not out.exists()
shutil.copytree(D/'phase3/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill_transport.cu'
s=p.read_text().replace('(c*z-d*v)','__fmaf_rn(c,z,-(d*v))').replace('(d*u-b*z)','__fmaf_rn(d,u,-(b*z))').replace('(b*v-c*u)','__fmaf_rn(b,v,-(c*u))')
p.write_text(s)
