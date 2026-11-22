"""Teacher增量候选：有界64D冻结投影、四Head并行及当前CPU位姿直入。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'teacher_plus_source';shutil.copytree(HERE/'teacher_tensor_source',root)
    (HERE/'teacher_plus_build').mkdir();shutil.copy2(HERE/'teacher_parallel.cuh',root/'teacher_parallel.cuh')
    path=root/'tensor_networks.cuh';path.write_text(path.read_text()+'\n#include "teacher_parallel.cuh"\n')
    path=root/'teacher_tensor.cuh';s=path.read_text().replace('template<int Mode,int Warps,bool Direct=false>',
        'template<int Mode,int Warps,bool Direct=false,bool Projected=false>')
    start=s.index('  for(int i=tid;i<Points*51');end=s.index('  if(warp<Warps)tensorMultiply',start)
    physical=s[start:end];physical=physical[:physical.rfind('  __syncthreads();')]
    projected='''  if constexpr(Projected){
    const float* fixed=static_cast<const float*>(packed.p[5]);const float* dynamic=static_cast<const float*>(packed.p[7]);
    for(int i=tid;i<Points*64;i+=Threads){const int p=i/64,c=i%64;float value=0.f;
      if(begin+p<n)value=__fadd_rn(fixed[ids[p]*64+c],dynamic[anchors[p]*64+c]);
      hidden[p][c]=Math::convert(relu(__fadd_rn(value,w.p[1][c])));}
  }else{
'''+physical+'''  }
  __syncthreads();
'''
    path.write_text(s[:start]+projected+s[end:])
    path=root/'deployment_prepare.cpp';s=path.read_text().replace('compensator_mode<=4','compensator_mode<=8')
    s=s.replace('mycartgs.pending-raster.shared.teacher.v1','mycartgs.pending-raster.shared.teacherplus.v1')
    anchor='TORCH_CHECK(compensator_.size()==13,"Projected input requires frozen 32/32 student");'
    assert anchor in s;s=s.replace(anchor,'''TORCH_CHECK(compensator_.size()==13||compensator_.size()==21,"Projected frozen network shape");
      const int64_t hidden=compensator_.size()==13?32:64;
      const int64_t cache_bytes=4*hidden*(total_+int64_t(tables_.size())*state_[0].size(0));
      TORCH_CHECK(cache_bytes<=256ll*1024*1024,"Frozen projection capacity limit: 256MiB");''')
    s=s.replace('torch::empty({state_[0].size(0),32},state_[0].options())','torch::empty({state_[0].size(0),hidden},state_[0].options())')
    pose=(HERE/'joint_source_pose/deployment_prepare.cpp').read_text()
    start=pose.index('  pybind11::tuple render_pose_current(');end=pose.index('  torch::Tensor debug_control() const',start)
    s=s.replace('#include <torch/extension.h>','#include <torch/extension.h>\n#include <pybind11/numpy.h>')
    s=s.replace('class TensorDeploymentPipeline {','std::vector<torch::Tensor> preparation_camera_parts(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);\nclass TensorDeploymentPipeline {')
    s=s.replace('  torch::Tensor debug_control() const',pose[start:end]+'  torch::Tensor debug_control() const')
    anchor='    .def("_render_pending_current",&TensorDeploymentPipeline::render_pending_current)'
    s=s.replace(anchor,anchor+'\n    .def("_render_pose_current",&TensorDeploymentPipeline::render_pose_current)');path.write_text(s)
    path=root/'raster/graph_main.cu';s=path.read_text().replace('TeacherPendingNineRasterizer','TeacherPlusPendingNineRasterizer')
    path.write_text(s.replace('mycartgs.pending-raster.shared.teacher.v1','mycartgs.pending-raster.shared.teacherplus.v1'))
    path=root/'deployment_runtime_kernels.cu';s=path.read_text().replace('compensation_mode<=4','compensation_mode<=8')
    s=s.replace('#include <torch/extension.h>','#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>')
    s=s.replace('mode=mode==3?2:1','mode=(mode&1)?2:1')
    anchor='''    preparationProjectAnchors<<<(anchors*32+127)/128,128,0,stream>>>'''
    start=s.index(anchor);end=s.index('  }\n  const auto rows=',start)
    original=s[start:end]
    new=original.replace('preparationProjectAnchors<<<(anchors*32','preparationProjectTeacherAnchors<<<(anchors*64')
    s=s[:start]+'''    if(compensation.size()==21){
'''+new+'''    }else{
'''+original+'''    }
'''+s[end:]
    start=s.index('      if(compensation_mode>=3){');end=s.index('      } else if(compensation_mode) {',start)
    branch='''      if(compensation_mode>=3){
        TORCH_CHECK(compensation.size()==21&&(tensor_weights.size()==5||tensor_weights.size()==8),"Teacher Tensor snapshot");
        TensorWeights tw{};for(int i=1;i<4;++i)tw.p[i]=tensor_weights[i].data_ptr();
        if(tensor_weights.size()==8){tw.p[5]=tensor_weights[5].data_ptr();tw.p[7]=tensor_weights[7].data_ptr();}
        auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
          auto kernel=[&](auto projected){constexpr bool Projected=decltype(projected)::value;
            const int64_t parallel_blocks=compensation_mode>=7?std::min<int64_t>((capacity+15)/16,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount):(capacity+15)/16;
            if(compensation_mode>=5)compensationTeacherParallel<Mode,Direct,Projected><<<parallel_blocks,128,0,stream>>>(
                capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),
                t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,
                m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
            else compensationTeacherTensor<Mode,Warps,Direct,Projected><<<(capacity+Warps*16-1)/(Warps*16),Warps*128,0,stream>>>(
                capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),
                t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,
                m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
          };if(tensor_weights.size()==8)kernel(std::true_type{});else kernel(std::false_type{});
        };dispatchTensor((compensation_mode&1)?2:1,tensor_comp_warps,launch);
'''
    s=s[:start]+branch+s[end:];path.write_text(s)
    path=root/'tensor_components.inc';s=path.read_text();start=s.index('std::vector<torch::Tensor> deployment_tensor_compensate(')
    prefix,component=s[:start],s[start:];component=component.replace('mode<=4','mode<=8')
    component=component.replace('(mode==1||mode==4)?at::kFloat:at::kHalf','(mode==1||(mode>=4&&!(mode&1)))?at::kFloat:at::kHalf')
    component=component.replace('mode>=3?(mode==3?2:1):mode','mode>=3?((mode&1)?2:1):mode')
    component=component.replace('  if(n){auto launch=', '  const bool parallel_teacher=mode>=5,persistent_teacher=mode>=7;\n  if(n){auto launch=')
    line=next(line for line in component.splitlines() if 'if(packed.size()==8)preparationProjectAnchors<<<' in line)
    teacher=line.replace('preparationProjectAnchors','preparationProjectTeacherAnchors').replace('size(0)*32','size(0)*64')
    teacher=teacher.replace('if(packed.size()==8)','if(packed.size()==8&&weights.size()==21)')
    student=line.replace('if(packed.size()==8)','else if(packed.size()==8)')
    component=component.replace(line,teacher+'\n'+student)
    start=component.index('    if(weights.size()==21){');end=component.index('    auto kernel=[&](auto projected)',start)
    branch='''    if(weights.size()==21){
      auto kernel=[&](auto projected){constexpr bool Projected=decltype(projected)::value;
        const int64_t blocks=persistent_teacher?std::min<int64_t>((n+15)/16,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount):(n+15)/16;
        if(parallel_teacher)compensationTeacherParallel<Mode,false,Projected><<<blocks,128,0,c10::cuda::getCurrentCUDAStream()>>>(
            n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,
            view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,
            output.data_ptr<float>(),residual.data_ptr<float>(),status.data_ptr<int>(),nullptr,nullptr,tw);
        else compensationTeacherTensor<Mode,Warps,false,Projected><<<(n+Warps*16-1)/(Warps*16),Warps*128,0,c10::cuda::getCurrentCUDAStream()>>>(
            n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,
            view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,
            output.data_ptr<float>(),residual.data_ptr<float>(),status.data_ptr<int>(),nullptr,nullptr,tw);
      };if(packed.size()==8)kernel(std::true_type{});else kernel(std::false_type{});
      return;
    }
'''
    component=component[:start]+branch+component[end:];path.write_text(prefix+component)
    print('TEACHER_PLUS_SOURCE_READY','64D bound=256MiB; independent projection/parallel flags; current pose path')


if __name__=='__main__':main()
