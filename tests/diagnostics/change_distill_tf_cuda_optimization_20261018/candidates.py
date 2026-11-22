"""从冻结 baseline 生成候选；不触碰正式工作树。"""
from pathlib import Path
import shutil
import sys

D=Path(__file__).resolve().parent
name=sys.argv[1]
out=D/name
assert not out.exists()
shutil.copytree(D/'baseline/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill.cpp'
s=p.read_text()
start=s.index('// Resolve membership')
end=s.index('std::array<torch::Tensor, 3> tfErrors')
s=s[:start]+'''// Event-local batch plans preserve the original group order, GEMM shape and rows.
struct TransportBatchPlan {
  struct Segment { size_t group; int64_t begin, end; };
  torch::Tensor positions;
  std::vector<Segment> segments;
};

class TargetTransport {
 public:
  explicit TargetTransport(const ChangeEvent& event) : groups_(event.correction_groups) {
    if (!groups_.empty()) group_ids_ = event.correction_group_id.cpu().contiguous();
  }

  // Called once for each initial evaluation population, before the ratio loop.
  // One population D2H and one packed H2D, never a per-miss roundtrip.
  std::vector<TransportBatchPlan> prepare(const torch::Tensor& indices, int64_t batch_size) const {
    std::vector<TransportBatchPlan> plans((indices.numel() + batch_size - 1) / batch_size);
    if (groups_.empty() || indices.numel() == 0) return plans;
    const auto host = indices.cpu().contiguous();
    const auto* ids = host.data_ptr<int64_t>();
    const auto* groups = group_ids_.data_ptr<int64_t>();
    std::vector<int64_t> packed;
    packed.reserve(indices.numel());
    for (size_t b = 0; b < plans.size(); ++b) {
      const auto begin = static_cast<int64_t>(b) * batch_size;
      const auto end = std::min(begin + batch_size, indices.numel());
      std::vector<std::vector<int64_t>> local(groups_.size());
      for (int64_t i = begin; i < end; ++i) {
        const auto g = groups[ids[i]];
        if (g >= 0) local[g].push_back(i - begin);
      }
      for (size_t g = 0; g < local.size(); ++g) {
        if (local[g].empty()) continue;
        const auto offset = static_cast<int64_t>(packed.size());
        packed.insert(packed.end(), local[g].begin(), local[g].end());
        plans[b].segments.push_back({g, offset, static_cast<int64_t>(packed.size())});
      }
    }
    if (!packed.empty()) {
      const auto positions = torch::tensor(packed, torch::kLong).to(indices.device());
      for (auto& plan : plans) plan.positions = positions;
    }
    return plans;
  }

  GaussianAttrState apply(GaussianAttrState target, const torch::Tensor& indices,
                          const TransportBatchPlan* plan = nullptr) const {
    if (groups_.empty() || indices.numel() == 0) return target;
    torch::NoGradGuard no_grad;
    std::vector<TransportBatchPlan> fallback;
    if (!plan) { fallback = prepare(indices, indices.numel()); plan = &fallback.front(); }
    for (const auto& segment : plan->segments) {
      const auto positions_g = plan->positions.slice(0, segment.begin, segment.end);
      const auto moved = applyGaussianTransform(target.select(positions_g), groups_[segment.group]);
      target = {target.xyz.index_copy(0, positions_g, moved.xyz),
                target.log_scaling.index_copy(0, positions_g, moved.log_scaling),
                target.rotation.index_copy(0, positions_g, moved.rotation)};
    }
    return target;
  }

 private:
  const std::vector<CorrectionGroup>& groups_;
  torch::Tensor group_ids_;
};

GaussianAttrState transportTarget(GaussianAttrState target, const torch::Tensor& indices,
                                  const TargetTransport& transport,
                                  const TransportBatchPlan* plan = nullptr) {
  return transport.apply(std::move(target), indices, plan);
}

'''+s[end:]
s=s.replace('bool fully_cached = false) {\n      const auto ratio', 'bool fully_cached = false, const TransportBatchPlan* plan = nullptr) {\n      const auto ratio')
s=s.replace('ratio), missing, transport);', 'ratio), missing, transport,\n            missing.numel() == batch.numel() ? plan : nullptr);')
token='''      if (indices.numel() == 0) return sums;
      for (int64_t begin = 0; begin < indices.numel(); begin += batch_size) {'''
assert s.count(token)==1
s=s.replace(token,'''      if (indices.numel() == 0) return sums;
      const auto plans = fully_cached ? std::vector<TransportBatchPlan>{}
                                     : transport.prepare(indices, batch_size);
      for (int64_t begin = 0; begin < indices.numel(); begin += batch_size) {''')
s=s.replace('auto loss = errors(batch, r, fully_cached);','auto loss = errors(batch, r, fully_cached,\n              fully_cached ? nullptr : &plans[begin / batch_size]);')
p.write_text(s)
print(out)
