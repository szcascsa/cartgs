"""独立Teacher Tensor Core候选，模式3=FP16隐藏层、4=TF32隐藏层。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'teacher_tensor_source';shutil.copytree(HERE/'joint_source_v2',root)
    (HERE/'teacher_tensor_build').mkdir();shutil.copy2(HERE/'teacher_tensor.cuh',root/'teacher_tensor.cuh')
    path=root/'tensor_networks.cuh';s=path.read_text()+'\n#include "teacher_tensor.cuh"\n';path.write_text(s)
    path=root/'deployment_prepare.cpp';s=path.read_text().replace('compensator_mode<=2','compensator_mode<=4')
    s=s.replace('Tensor mode: 0=CUDA,1=TF32,2=FP16','Tensor mode: 0=CUDA,1/2=Student TF32/FP16,3/4=Teacher FP16/TF32')
    s=s.replace('mycartgs.pending-raster.shared.v2','mycartgs.pending-raster.shared.teacher.v1');path.write_text(s)
    path=root/'raster/graph_main.cu';s=path.read_text().replace('NativePendingNineRasterizer','TeacherPendingNineRasterizer')
    s=s.replace('mycartgs.pending-raster.shared.v2','mycartgs.pending-raster.shared.teacher.v1');path.write_text(s)
    path=root/'deployment_runtime_kernels.cu';s=path.read_text()
    anchor='TORCH_CHECK(!compensation_mode||student.size()==13,"Tensor Compensation requires a 32/32 student");'
    assert anchor in s;s=s.replace(anchor,'''TORCH_CHECK(!compensation_mode ||
      (compensation_mode<=2&&student.size()==13) ||
      (compensation_mode>=3&&compensation_mode<=4&&student.size()==21),"Tensor Compensation snapshot/mode mismatch");''')
    anchor='''    int mode=(which==0||which==4)?selector_mode:compensation_mode;if(!mode)continue;
    int size=which==1?64*32:which==3||which==4?32*16:32*32;'''
    assert anchor in s;s=s.replace(anchor,'''    int mode=(which==0||which==4)?selector_mode:compensation_mode;if(!mode)continue;
    const bool teacher=mode>=3;
    if(teacher)mode=mode==3?2:1;
    int size=teacher?(which==3?4*64*64:64*64):(which==1?64*32:which==3||which==4?32*16:32*32);''')
    anchor='''    if(mode==1)packTensorWeights<1>'''
    inserted='''    if(teacher){
      if(mode==1)packTeacherTensorWeights<1><<<(size+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(cw,which,result[which].data_ptr<float>());
      else packTeacherTensorWeights<2><<<(size+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(cw,which,reinterpret_cast<__half*>(result[which].data_ptr<at::Half>()));
      continue;
    }
'''
    assert s.count(anchor)==1;s=s.replace(anchor,inserted+anchor)
    anchor='''      if(compensation_mode) {
        TORCH_CHECK(compensation.size()==13,"Tensor Compensation requires the frozen 32/32 student");'''
    teacher='''      if(compensation_mode>=3){
        TORCH_CHECK(compensation.size()==21&&tensor_weights.size()==5,"Teacher Tensor snapshot");
        TensorWeights tw{};for(int i=1;i<4;++i)tw.p[i]=tensor_weights[i].data_ptr();
        auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
          compensationTeacherTensor<Mode,Warps,Direct><<<(capacity+Warps*16-1)/(Warps*16),Warps*128,0,stream>>>(
              capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),
              t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,
              m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
        };dispatchTensor(compensation_mode==3?2:1,tensor_comp_warps,launch);
      } else if(compensation_mode) {
        TORCH_CHECK(compensation.size()==13,"Tensor Compensation requires the frozen 32/32 student");'''
    assert anchor in s;s=s.replace(anchor,teacher)
    path.write_text(s)
    path=root/'tensor_components.inc';s=path.read_text();begin=s.index('std::vector<torch::Tensor> deployment_tensor_compensate(')
    prefix,component=s[:begin],s[begin:]
    component=component.replace('weights.size()==13&&attributes.size()==6','(weights.size()==13||weights.size()==21)&&attributes.size()==6')
    component=component.replace('(mode==1||mode==2)','(mode>=1&&mode<=4)')
    component=component.replace('for(int j=0;j<13;++j)','for(size_t j=0;j<weights.size();++j)')
    component=component.replace('mode==1?at::kFloat:at::kHalf','(mode==1||mode==4)?at::kFloat:at::kHalf')
    anchor='''    auto kernel=[&](auto projected){constexpr bool Projected=decltype(projected)::value;'''
    assert anchor in component
    component=component.replace(anchor,'''    if(weights.size()==21){
      TORCH_CHECK(packed.size()==5,"Teacher component uses FP32 physical input");
      compensationTeacherTensor<Mode,Warps,false><<<(n+Warps*16-1)/(Warps*16),Warps*128,0,c10::cuda::getCurrentCUDAStream()>>>(
          n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,
          view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,
          output.data_ptr<float>(),residual.data_ptr<float>(),status.data_ptr<int>(),nullptr,nullptr,tw);
      return;
    }
'''+anchor)
    component=component.replace('dispatchTensor(mode,warps,launch);','dispatchTensor(mode>=3?(mode==3?2:1):mode,warps,launch);')
    path.write_text(prefix+component)
    print('TEACHER_TENSOR_SOURCE_READY','FP32 input/output; five hidden 64x64 MMA; independent heads')


if __name__=='__main__':main()
