from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase2'
assert not out.exists()
shutil.copytree(D/'phase1/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
shutil.copy2(D/'transport_cuda.h',out/'source/include/change_distill_transport_cuda.h')
shutil.copy2(D/'transport_cuda.cu',out/'source/src/change_distill_transport.cu')
p=out/'source/src/change_distill.cpp'
s=p.read_text().replace('#include "include/transform_field_cuda_optimizer.h"','#include "include/transform_field_cuda_optimizer.h"\n#include "include/change_distill_transport_cuda.h"')
token='''    for (const auto& segment : plan->segments) {
      const auto positions_g'''
replacement='''    if (plan->segments.empty()) return target;
    if (target.xyz.is_cuda() && target.xyz.scalar_type() == torch::kFloat32) {
      GaussianAttrState output{target.xyz.contiguous().clone(),
          target.log_scaling.contiguous().clone(), target.rotation.contiguous().clone()};
      for (const auto& segment : plan->segments) {
        const auto positions = plan->positions.slice(0, segment.begin, segment.end);
        const auto& correction = groups_[segment.group];
        torch::Tensor product;
        if (correction.changed_attributes & toBits(GaussianAttrFlag::XYZ)) {
          const auto scaled = cd_transport_cuda::gatherScaledXYZ(target.xyz, positions, correction.scale);
          // Keep the exact original group GEMM: Mx3, same row order and matrix view.
          product = torch::matmul(scaled, correction.transform_4x4.slice(0, 0, 3).slice(1, 0, 3));
        }
        cd_transport_cuda::scatter(target, output, positions, product, correction);
      }
      return output;
    }
    for (const auto& segment : plan->segments) {
      const auto positions_g'''
assert s.count(token)==1
s=s.replace(token,replacement)
p.write_text(s)
