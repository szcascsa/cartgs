"""只在已有 nonzero 结果证明全部 in-domain 时省去 identity gather/scatter。"""
from pathlib import Path
import shutil,sys
D=Path(__file__).resolve().parent;base,name=sys.argv[1:3];out=D/name
assert not out.exists();shutil.copytree(D/base/'source',out/'source');(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']:shutil.copy2(D/base/f,out/f)
p=out/'source/src/transform_field.cpp';s=p.read_text(encoding='utf-8')
start=s.index('std::vector<GaussianAttrState> TransformFieldImpl::applyLogResidualMany(')
end=s.find('\nstd::vector<TransformResidual>',start)
if end<0:end=len(s)
part=s[start:end]
assert part.count('  auto selected = state.select(indices);')==1
part=part.replace('  auto selected = state.select(indices);','''  const bool all_inside = indices.numel() == state.xyz.size(0);
  auto selected = all_inside ? state : state.select(indices);''')
token='''    result.push_back({state.xyz.index_copy(0, indices, output[0]),'''
assert part.count(token)==1
part=part.replace(token,'''    if (all_inside) {
      result.push_back({output[0], output[1], output[2]});
      continue;
    }
'''+token)
s=s[:start]+part+s[end:];p.write_text(s,encoding='utf-8')
