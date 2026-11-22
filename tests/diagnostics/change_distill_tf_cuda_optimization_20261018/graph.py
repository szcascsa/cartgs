"""独立 event-local transport graph prototype；含捕获/实例化成本。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase4_graph'
assert not out.exists()
shutil.copytree(D/'phase3/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill.cpp'
s=p.read_text().replace('#include <algorithm>','#include <ATen/cuda/CUDAGraph.h>\n#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAStream.h>\n#include <algorithm>')
s=s.replace('struct TransportBatchPlan {','''struct TransportGraphState {
  at::cuda::CUDAGraph graph;
  GaussianAttrState input, output;
};
struct TransportBatchPlan {
  mutable std::shared_ptr<TransportGraphState> graph;''')
token='''  GaussianAttrState apply(GaussianAttrState target, const torch::Tensor& indices,
                          const TransportBatchPlan* plan = nullptr) const {'''
assert s.count(token)==1
s=s.replace(token,'''  GaussianAttrState apply(GaussianAttrState target, const torch::Tensor& indices,
                          const TransportBatchPlan* plan = nullptr) const {
    if (!plan || plan->segments.empty() || !target.xyz.is_cuda())
      return applyPlain(target,indices,plan);
    torch::NoGradGuard no_grad;
    if (!plan->graph) {
      auto state=std::make_shared<TransportGraphState>();
      state->input={target.xyz.clone(),target.log_scaling.clone(),target.rotation.clone()};
      // Prototype setup only. Included in the native event wall measurement.
      c10::cuda::getCurrentCUDAStream().synchronize();
      const auto side=c10::cuda::getStreamFromPool();
      {
        c10::cuda::CUDAStreamGuard guard(side);
        const auto warm=applyPlain(state->input,indices,plan);
        side.synchronize();
        state->graph.capture_begin();
        state->output=applyPlain(state->input,indices,plan);
        state->graph.capture_end();
      }
      plan->graph=std::move(state);
    }
    auto& state=*plan->graph;
    state.input.xyz.copy_(target.xyz);state.input.log_scaling.copy_(target.log_scaling);
    state.input.rotation.copy_(target.rotation);
    state.graph.replay();
    return state.output;
  }

  GaussianAttrState applyPlain(GaussianAttrState target, const torch::Tensor& indices,
                          const TransportBatchPlan* plan = nullptr) const {''')
p.write_text(s)
