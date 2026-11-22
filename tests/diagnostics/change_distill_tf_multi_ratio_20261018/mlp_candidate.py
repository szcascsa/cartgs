"""独立 multi-ratio MLP 实验；safe 保留原 B 行 backward。"""
from pathlib import Path
import shutil,sys
D=Path(__file__).resolve().parent
base,name,mode=sys.argv[1:4];out=D/name
assert not out.exists();shutil.copytree(D/base/'source',out/'source');(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']:shutil.copy2(D/base/f,out/f)
p=out/'source/src/transform_field.cpp';s=p.read_text(encoding='utf-8')
if 'TransformFieldImpl::forwardMany(' not in s:
 s=s.replace('''  for (auto ratio : ratios) {
    auto residual = forward(selected.xyz, ratio);''','''  auto residuals = forwardMany(selected.xyz, ratios);
  for (std::size_t r = 0; r < ratios.size(); ++r) {
    const auto& residual = residuals[r];''')
 template=(D/'a1_spatial/source/src/transform_field.cpp').read_text(encoding='utf-8')
 s+='\n'+template[template.index('std::vector<TransformResidual> TransformFieldImpl::forwardMany('):]
 s=s.replace('  auto features = tf_cuda::queryMany(xyz, ratios, aabbMin(), aabbMax(), gridParameters());','''  std::vector<torch::Tensor> features;
  for (auto ratio : ratios) {
    auto ratio_tensor = torch::full({xyz.size(0),1}, ratio, xyz.options());
    features.push_back(tf_cuda::query(xyz, ratio_tensor, aabbMin(), aabbMax(), gridParameters()));
  }''')
 h=out/'source/include/transform_field.h';hs=h.read_text(encoding='utf-8')
 hs=hs.replace('  ElasticGaussianAttributes applyResidual(','''  std::vector<TransformResidual> forwardMany(
      const torch::Tensor& xyz, const std::vector<float>& ratios);
  ElasticGaussianAttributes applyResidual(''');h.write_text(hs,encoding='utf-8')
start=s.index('  for (std::size_t r = 0; r < ratios.size(); ++r) {',s.index('TransformFieldImpl::forwardMany('))
end=s.index('  return result;',start)
s=s[:start]+'''  if (ratios.empty()) return result;
  for (std::size_t r = 0; r < ratios.size(); ++r) {
      const auto& feature = features[r];
      (void)feature;
  }
  const auto n = xyz.size(0);
  const auto mlp_backend = features[0].size(1) == 16 && config_.hidden_dim == 32 && n * ratios.size() <= 10000
      ? tf_cuda::MlpBackend::Warp32 : tf_cuda::MlpBackend::Cublas;
'''+('''  auto residual = tf_cuda::neural(torch::cat(features, 0), mlpParameters(), mlp_backend);
  for (std::size_t r = 0; r < ratios.size(); ++r)
    result.push_back({residual[0].narrow(0,r*n,n),residual[1].narrow(0,r*n,n),residual[2].narrow(0,r*n,n)});
''' if mode=='full' else '''  auto residual = tf_cuda::neuralMany(features, mlpParameters(), mlp_backend);
  for (const auto& x : residual) result.push_back({x[0],x[1],x[2]});
''')+s[end:]
p.write_text(s,encoding='utf-8')
if mode=='safe':
 p=out/'source/include/transform_field_cuda.h';p.write_text(p.read_text(encoding='utf-8')+'''
namespace tf_cuda {
std::vector<std::vector<torch::Tensor>> neuralMany(const std::vector<torch::Tensor>& features,
    const std::vector<torch::Tensor>& parameters, MlpBackend backend);
}
''',encoding='utf-8')
 p=out/'source/src/transform_field_cuda.cpp';s=p.read_text(encoding='utf-8')
 token='class OutputFunction :'
 s=s.replace(token,'''class CachedNeuralFunction : public torch::autograd::Function<CachedNeuralFunction> {
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
      at::TensorList inputs, at::TensorList context) {
    auto saved=inputs.vec();ctx->set_materialize_grads(false);
    saved.insert(saved.end(),context.begin()+1,context.begin()+5);
    ctx->save_for_backward(saved);captureProfile(ctx);
    return {context[5],context[6],context[7]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list gradients) {
    auto saved=ctx->get_saved_variables();
    ProtectionProfiler::Scope timing(capturedProfile(ctx),ProtectionProfiler::TFBackward,true);
    for(auto& g:gradients)if(g.defined())g=g.contiguous();
    std::vector<torch::Tensor> context{saved[0]};context.insert(context.end(),saved.begin()+15,saved.end());
    auto result=mlpBackward(context,{saved.begin()+1,saved.begin()+15},gradients);
    for(int i=0;i<8;++i)result.push_back(torch::Tensor());
    return result;
  }
};
'''+token,1)
 pos=s.rindex('}  // namespace tf_cuda')
 s=s[:pos]+'''std::vector<std::vector<torch::Tensor>> neuralMany(const std::vector<torch::Tensor>& features,
    const std::vector<torch::Tensor>& parameters,MlpBackend backend) {
  std::vector<std::vector<torch::Tensor>> result;
  if(features.empty())return result;
  auto context=mlpForward(torch::cat(features,0),parameters,backend);
  const auto n=features[0].size(0);
  for(size_t r=0;r<features.size();++r) {
    std::vector<torch::Tensor> part{features[r]},inputs{features[r]};
    for(int i=1;i<8;++i)part.push_back(context[i].narrow(0,r*n,n));
    inputs.insert(inputs.end(),parameters.begin(),parameters.end());
    result.push_back(CachedNeuralFunction::apply(at::TensorList(inputs),at::TensorList(part)));
  }
  return result;
}
'''+s[pos:];p.write_text(s,encoding='utf-8')
