"""Candidate 3: 单次 compact gather/scatter，逐组原 mm_out GEMM。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase3'
assert not out.exists()
shutil.copytree(D/'phase2_exact/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/include/change_distill_transport_cuda.h'
p.write_text(p.read_text()+'''
namespace cd_transport_cuda {
torch::Tensor compactXYZ(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
void scatterCompact(const GaussianAttrState&, const GaussianAttrState&,
                    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
}
''')
p=out/'source/src/change_distill_transport.cu'
p.write_text(p.read_text()+'''
namespace cd_transport_cuda {
namespace {
__global__ void compactXYZKernel(const float* x, int64_t s0, int64_t s1,
    const int64_t* pos, const int64_t* groups, const float* meta, int64_t n, float* out) {
  const int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*3) out[i]=x[pos[i/3]*s0+(i%3)*s1]*meta[groups[i/3]*23+20];
}
__global__ void scatterCompactKernel(ReadState in, float* x, float* s, float* q,
    const int64_t* pos, const int64_t* groups, const float* meta,
    const float* product, int64_t n) {
  const int64_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n) return;
  const auto p=pos[i]; const auto* m=meta+groups[i]*23;
  const auto flags=static_cast<unsigned>(m[22]);
  if(flags&1) for(int j=0;j<3;++j) x[p*3+j]=product[i*3+j]+m[12+j];
  if(flags&2) for(int j=0;j<3;++j) s[p*3+j]=in.s[p*in.ss0+j*in.ss1]+m[21];
  if(flags&4) {
    const float a=m[16], b=m[17], c=m[18], d=m[19];
    const float w=in.q[p*in.qs0], u=in.q[p*in.qs0+in.qs1];
    const float v=in.q[p*in.qs0+2*in.qs1], z=in.q[p*in.qs0+3*in.qs1];
    const float o0=a*w-((b*u+d*z)+c*v);
    const float o1=(a*u+w*b)+(c*z-d*v);
    const float o2=(a*v+w*c)+(d*u-b*z);
    const float o3=(a*z+w*d)+(b*v-c*u);
    const float norm=fmaxf(sqrtf((o0*o0+o1*o1)+(o2*o2+o3*o3)),1e-12f);
    q[p*4]=o0/norm;q[p*4+1]=o1/norm;q[p*4+2]=o2/norm;q[p*4+3]=o3/norm;
  }
}
}
torch::Tensor compactXYZ(const torch::Tensor& x,const torch::Tensor& pos,
                         const torch::Tensor& groups,const torch::Tensor& meta) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out=torch::empty({pos.numel(),3},x.options().requires_grad(false));
  if(pos.numel()) compactXYZKernel<<<(pos.numel()*3+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),x.stride(0),x.stride(1),pos.data_ptr<int64_t>(),groups.data_ptr<int64_t>(),meta.data_ptr<float>(),pos.numel(),out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
void scatterCompact(const GaussianAttrState& input,const GaussianAttrState& output,
                    const torch::Tensor& pos,const torch::Tensor& groups,
                    const torch::Tensor& meta,const torch::Tensor& product) {
  const c10::cuda::CUDAGuard guard(input.xyz.device());
  const ReadState in{input.xyz.data_ptr<float>(),input.log_scaling.data_ptr<float>(),input.rotation.data_ptr<float>(),
      input.xyz.stride(0),input.xyz.stride(1),input.log_scaling.stride(0),input.log_scaling.stride(1),input.rotation.stride(0),input.rotation.stride(1)};
  if(pos.numel()) scatterCompactKernel<<<(pos.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(
      in,output.xyz.data_ptr<float>(),output.log_scaling.data_ptr<float>(),output.rotation.data_ptr<float>(),pos.data_ptr<int64_t>(),groups.data_ptr<int64_t>(),meta.data_ptr<float>(),product.data_ptr<float>(),pos.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
''')
p=out/'source/src/change_distill.cpp'
s=p.read_text().replace('  torch::Tensor positions;','  torch::Tensor positions, group_indices;')
token='    if (!groups_.empty()) group_ids_ = event.correction_group_id.cpu().contiguous();'
s=s.replace(token,token+'''
    if (!groups_.empty() && event.correction_group_id.is_cuda()) {
      const auto options=event.correction_groups.front().indices.options().dtype(torch::kFloat32);
      std::vector<torch::Tensor> matrices, rotations;
      std::vector<float> constants;
      for (const auto& g : groups_) {
        matrices.push_back(g.transform_4x4.defined() ? g.transform_4x4.reshape({16}) : torch::empty({16},options));
        rotations.push_back(g.rotation_wxyz.defined() ? g.rotation_wxyz.reshape({4}) : torch::empty({4},options));
        constants.insert(constants.end(),{g.scale,std::log(g.scale),static_cast<float>(g.changed_attributes)});
      }
      metadata_=torch::cat({torch::stack(matrices),torch::stack(rotations),
          torch::tensor(constants,torch::kFloat32).reshape({static_cast<int64_t>(groups_.size()),3}).to(options.device())},1).contiguous();
    }''')
s=s.replace('    std::vector<int64_t> packed;','    std::vector<int64_t> packed, packed_groups;')
s=s.replace('        packed.insert(packed.end(), local[g].begin(), local[g].end());','        packed.insert(packed.end(), local[g].begin(), local[g].end());\n        packed_groups.insert(packed_groups.end(),local[g].size(),g);')
s=s.replace('      const auto positions = torch::tensor(packed, torch::kLong).to(indices.device());\n      for (auto& plan : plans) plan.positions = positions;', '''      const auto count=static_cast<int64_t>(packed.size());
      packed.insert(packed.end(),packed_groups.begin(),packed_groups.end());
      const auto data=torch::tensor(packed,torch::kLong).reshape({2,count}).to(indices.device());
      for (auto& plan : plans) { plan.positions=data[0]; plan.group_indices=data[1]; }''')
begin=s.index('      for (const auto& segment : plan->segments) {')
end=s.index('      return output;',begin)
s=s[:begin]+'''      const auto base=plan->segments.front().begin, end=plan->segments.back().end;
      const auto positions=plan->positions.slice(0,base,end);
      const auto groups=plan->group_indices.slice(0,base,end);
      const auto scaled=cd_transport_cuda::compactXYZ(target.xyz,positions,groups,metadata_);
      auto product=torch::empty_like(scaled);
      for (const auto& segment : plan->segments) {
        const auto& correction=groups_[segment.group];
        if (correction.changed_attributes & toBits(GaussianAttrFlag::XYZ)) {
          auto result=product.slice(0,segment.begin-base,segment.end-base);
          torch::mm_out(result,scaled.slice(0,segment.begin-base,segment.end-base),
              correction.transform_4x4.slice(0,0,3).slice(1,0,3));
        }
      }
      cd_transport_cuda::scatterCompact(target,output,positions,groups,metadata_,product);
'''+s[end:]
s=s.replace('  torch::Tensor group_ids_;','  torch::Tensor group_ids_, metadata_;')
p.write_text(s)
