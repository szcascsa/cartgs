"""共享 XY/XZ/YZ 采样；保留六因子原乘法顺序和四个原 backward。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent;out=D/'a1_spatial'
assert not out.exists();shutil.copytree(D/'a0_prepare/source',out/'source');(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']:shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/include/transform_field_cuda.h';s=p.read_text(encoding='utf-8')
s+='''
namespace tf_cuda {
std::vector<torch::Tensor> queryMany(const torch::Tensor& xyz,
    const std::vector<float>& ratios, const torch::Tensor& lo, const torch::Tensor& hi,
    const std::vector<torch::Tensor>& planes);
torch::Tensor queryManyForward(const std::vector<torch::Tensor>& inputs,
                               const std::vector<float>& ratios);
}
''';p.write_text(s,encoding='utf-8')
p=out/'source/src/transform_field_cuda.cu';s=p.read_text(encoding='utf-8')
token='__global__ void queryGradKernel('
insert='''struct Ratios { float values[4]; int count; };
__global__ void queryManyKernel(Inputs in, Planes p, Ratios ratios, float* out, int n) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  const int f=p.channels*(p.count/6);
  if(index>=n*f) return;
  const int row=index/f,k=index%f,level=k/p.channels,ch=k%p.channels;
  float c[4];coordinates(in,row,c);
  const float xy=sample(p,level*6,ch,c[0],c[1]).value;
  const float xz=sample(p,level*6+1,ch,c[0],c[2]).value;
  const float yz=sample(p,level*6+3,ch,c[1],c[2]).value;
  for(int r=0;r<ratios.count;++r) {
    c[3]=ratios.values[r];
    float value=1.0f;
    // Do not group the three spatial factors: XR is between XZ and YZ.
    value*=xy;value*=xz;
    value*=sample(p,level*6+2,ch,c[0],c[3]).value;
    value*=yz;
    value*=sample(p,level*6+4,ch,c[1],c[3]).value;
    value*=sample(p,level*6+5,ch,c[2],c[3]).value;
    out[r*n*f+index]=value;
  }
}
'''
assert s.count(token)==1;s=s.replace(token,insert+token)
pos=s.rindex('}  // namespace tf_cuda')
s=s[:pos]+'''torch::Tensor queryManyForward(const std::vector<torch::Tensor>& inputs,
                               const std::vector<float>& ratios) {
  auto pack=describe(inputs);c10::cuda::CUDAGuard guard(inputs[0].device());
  TORCH_CHECK(!ratios.empty() && ratios.size()<=4,"queryMany supports up to four ratios");
  const int n=inputs[0].size(0),f=pack.channels*(pack.count/6);
  auto out=torch::empty({static_cast<int64_t>(ratios.size()),n,f},inputs[0].options().requires_grad(false));
  Ratios r{};r.count=ratios.size();for(size_t i=0;i<ratios.size();++i)r.values[i]=ratios[i];
  if(n) {
    queryManyKernel<<<(n*f+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(describeInputs(inputs),pack,r,out.data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
'''+s[pos:];p.write_text(s,encoding='utf-8')
p=out/'source/src/transform_field_cuda.cpp';s=p.read_text(encoding='utf-8')
token='class NeuralFunction :'
s=s.replace(token,'''class CachedQueryFunction : public torch::autograd::Function<CachedQueryFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               at::TensorList inputs, torch::Tensor feature) {
    ctx->save_for_backward(inputs.vec());captureProfile(ctx);return feature;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                torch::autograd::variable_list gradients) {
    ProtectionProfiler::Scope timing(capturedProfile(ctx), ProtectionProfiler::TFBackward, true);
    auto result=queryBackward(ctx->get_saved_variables(),gradients.at(0).contiguous());
    result.push_back(torch::Tensor());return result;
  }
};
'''+token,1)
pos=s.rindex('}  // namespace tf_cuda')
s=s[:pos]+'''std::vector<torch::Tensor> queryMany(const torch::Tensor& xyz,
    const std::vector<float>& ratios,const torch::Tensor& lo,const torch::Tensor& hi,
    const std::vector<torch::Tensor>& planes) {
  std::vector<torch::Tensor> result;
  if(ratios.empty())return result;
  if(ratios.size()==1 || ratios.size()>4) {
    for(auto r:ratios)result.push_back(query(xyz,torch::full({xyz.size(0),1},r,xyz.options()),lo,hi,planes));
    return result;
  }
  std::vector<std::vector<torch::Tensor>> inputs;
  for(auto r:ratios) {
    inputs.push_back({xyz,torch::full({xyz.size(0),1},r,xyz.options()),lo,hi});
    inputs.back().insert(inputs.back().end(),planes.begin(),planes.end());
  }
  auto features=queryManyForward(inputs.front(),ratios);
  for(size_t r=0;r<ratios.size();++r)
    result.push_back(xyz.size(0)==0 ? features[r]
        : CachedQueryFunction::apply(at::TensorList(inputs[r]),features[r]));
  return result;
}
'''+s[pos:];p.write_text(s,encoding='utf-8')
p=out/'source/include/transform_field.h';s=p.read_text(encoding='utf-8')
s=s.replace('  ElasticGaussianAttributes applyResidual(','''  std::vector<TransformResidual> forwardMany(
      const torch::Tensor& xyz, const std::vector<float>& ratios);
  ElasticGaussianAttributes applyResidual(''');p.write_text(s,encoding='utf-8')
p=out/'source/src/transform_field.cpp';s=p.read_text(encoding='utf-8')
s=s.replace('''  for (auto ratio : ratios) {
    auto residual = forward(selected.xyz, ratio);''','''  auto residuals = forwardMany(selected.xyz, ratios);
  for (std::size_t r = 0; r < ratios.size(); ++r) {
    const auto& residual = residuals[r];''')
s+='''
std::vector<TransformResidual> TransformFieldImpl::forwardMany(
    const torch::Tensor& xyz, const std::vector<float>& ratios) {
  std::vector<TransformResidual> result;
  if (config_.backend != "cuda") {
    for (auto ratio : ratios) result.push_back(forward(xyz, ratio));
    return result;
  }
  auto features = tf_cuda::queryMany(xyz, ratios, aabbMin(), aabbMax(), gridParameters());
  for (std::size_t r = 0; r < ratios.size(); ++r) {
      const auto& feature = features[r];
      const auto mlp_backend = feature.size(1) == 16 && config_.hidden_dim == 32 && xyz.size(0) <= 10000
          ? tf_cuda::MlpBackend::Warp32 : tf_cuda::MlpBackend::Cublas;
      auto residual = tf_cuda::neural(feature, mlpParameters(), mlp_backend);
      result.push_back({residual[0], residual[1], residual[2]});
  }
  return result;
}
''';p.write_text(s,encoding='utf-8')
