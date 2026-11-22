"""evaluation 复用与延迟 scalar readback 分开验收。"""
from pathlib import Path
import shutil,sys
D=Path(__file__).resolve().parent;base,name,mode=sys.argv[1:4];out=D/name
assert not out.exists();shutil.copytree(D/base/'source',out/'source');(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']:shutil.copy2(D/base/f,out/f)
p=out/'source/src/change_distill.cpp';s=p.read_text(encoding='utf-8')
start=s.index('    auto evaluate =',s.index('    const TargetTransport transport(event);'))
end=s.index('    const auto pre =',start);part=s[start:end]
if mode=='reuse':
 token='''        for (std::size_t r = 0; r < context.ratios.size(); ++r) {'''
 assert part.count(token)==1
 part=part.replace(token,'''        auto predictions = context.tf->applyLogResidualMany(current.select(batch), context.ratios);
'''+token)
 part=part.replace('fully_cached ? nullptr : &plans[begin / batch_size]);','fully_cached ? nullptr : &plans[begin / batch_size], &predictions[r]);')
else:
 token='      std::array<double, 3> sums{};';assert part.count(token)==1
 part=part.replace(token,token+'''
      std::vector<torch::Tensor> scalars;
      std::vector<int64_t> scalar_counts;''')
 token='          for (int j = 0; j < 3; ++j) sums[j] += loss[j].item<double>() * batch.numel();'
 assert part.count(token)==1
 part=part.replace(token,'''          for (int j = 0; j < 3; ++j) {
            scalars.push_back(loss[j]);
            scalar_counts.push_back(batch.numel());
          }''')
 token='      for (auto& sum : sums) sum /= indices.numel() * ratio_count;'
 assert part.count(token)==1
 part=part.replace(token,'''      // Preserve FP32 reductions, then the original ordered CPU double fold.
      auto values = torch::stack(scalars).to(torch::kCPU).to(torch::kFloat64);
      const auto* data = values.data_ptr<double>();
      for (std::size_t i = 0; i < scalars.size(); ++i)
        sums[i % 3] += data[i] * scalar_counts[i];
'''+token)
s=s[:start]+part+s[end:];p.write_text(s,encoding='utf-8')
