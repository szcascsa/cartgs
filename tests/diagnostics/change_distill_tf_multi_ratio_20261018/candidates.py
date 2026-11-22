"""逐阶段独立快照，正式项目在所有 gate 后才更新。"""
from pathlib import Path
import shutil,sys
D=Path(__file__).resolve().parent
name=sys.argv[1];parent=sys.argv[2] if len(sys.argv)>2 else 'baseline'
out=D/name;assert not out.exists()
shutil.copytree(D/parent/'source',out/'source');(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']:shutil.copy2(D/'baseline'/f,out/f)
if name=='a0_prepare':
    p=out/'source/include/transform_field.h';s=p.read_text(encoding='utf-8')
    s=s.replace('  GaussianAttrState applyLogResidual(const GaussianAttrState& state, float ratio);','''  GaussianAttrState applyLogResidual(const GaussianAttrState& state, float ratio);
  std::vector<GaussianAttrState> applyLogResidualMany(
      const GaussianAttrState& state, const std::vector<float>& ratios);''');p.write_text(s,encoding='utf-8')
    p=out/'source/src/transform_field.cpp';s=p.read_text(encoding='utf-8')
    s+='''
std::vector<GaussianAttrState> TransformFieldImpl::applyLogResidualMany(
    const GaussianAttrState& state, const std::vector<float>& ratios) {
  std::vector<GaussianAttrState> result;
  if (ratios.empty()) return result;
  if (config_.backend != "cuda") {
    for (auto ratio : ratios) result.push_back(applyLogResidual(state, ratio));
    return result;
  }
  auto indices = config_.outside_identity
      ? torch::nonzero(insideMask(state.xyz)).reshape({-1})
      : torch::arange(state.xyz.size(0), state.xyz.options().dtype(torch::kLong));
  if (indices.numel() == 0) return std::vector<GaussianAttrState>(ratios.size(), state);
  auto selected = state.select(indices);
  for (auto ratio : ratios) {
    auto residual = forward(selected.xyz, ratio);
    auto output = tf_cuda::output(selected.xyz, selected.log_scaling, selected.rotation,
        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw}, true);
    result.push_back({state.xyz.index_copy(0, indices, output[0]),
                     state.log_scaling.index_copy(0, indices, output[1]),
                     state.rotation.index_copy(0, indices, output[2])});
  }
  return result;
}
''';p.write_text(s,encoding='utf-8')
    p=out/'source/src/change_distill.cpp';s=p.read_text(encoding='utf-8')
    s=s.replace('const TransportBatchPlan* plan = nullptr) {\n      const auto ratio', 'const TransportBatchPlan* plan = nullptr,\n                      const GaussianAttrState* prediction = nullptr) {\n      const auto ratio')
    s=s.replace('return tfErrors(context.tf->applyLogResidual(current.select(batch), ratio), target, diagonal);','return tfErrors(prediction ? *prediction : context.tf->applyLogResidual(current.select(batch), ratio), target, diagonal);')
    token='''          auto total = torch::zeros({}, current.xyz.options());
          for (std::size_t r = 0; r < context.ratios.size(); ++r) {'''
    assert s.count(token)==1
    s=s.replace(token,'''          auto predictions = context.tf->applyLogResidualMany(current.select(batch), context.ratios);
          auto total = torch::zeros({}, current.xyz.options());
          for (std::size_t r = 0; r < context.ratios.size(); ++r) {''')
    s=s.replace('auto loss = errors(batch, r, true);','auto loss = errors(batch, r, true, nullptr, &predictions[r]);')
    p.write_text(s,encoding='utf-8')
print(out)
