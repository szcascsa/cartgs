"""独立准备候选：每帧重算，复用有界相机缓冲；可选融合当前View和Student投影。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_preparation'
    shutil.copytree(HERE / 'source', root)
    (HERE / 'build_preparation').mkdir()
    shutil.copy2(HERE / 'student_view_project.cuh', root / 'student_view_project.cuh')
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text()
    anchor = '__global__ void selectorPrefix'
    text = text.replace(anchor, '#include "student_view_project.cuh"\n\n' + anchor)
    text = text.replace('if(local && schema!=1 && anchors)viewKernel', 'if(local && schema!=1 && anchors && compensation_mode!=9)viewKernel')
    text = text.replace('if(compensate && compensation_mode && tensor_weights.size()==8){', 'if(compensate && compensation_mode && tensor_weights.size()==8 && compensation_mode!=9){')
    anchor = '  const auto rows=buffers[1].data_ptr<int64_t>();'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''  if(compensation_mode==9){
    TORCH_CHECK(local&&schema!=1&&tensor_weights.size()==8,"Fused Student view requires local frozen projection");
    for(size_t j=0;j<compensation.size();++j)cw.p[j]=compensation[j].data_ptr<float>();
    studentViewProject<<<(anchors*32+127)/128,128,0,stream>>>(anchors,state[0].data_ptr<float>(),state[10].data_ptr<float>(),state[11].data_ptr<float>(),camera.data_ptr<float>(),vm.data_ptr<float>(),vm.stride(0),vm.stride(1),pm.data_ptr<float>(),pm.stride(0),pm.stride(1),state[12].data_ptr<float>(),aspect,near,buffers[2].data_ptr<float>(),frame,destination,visible.defined()?visible.data_ptr<bool>():nullptr,tensor_weights[6].data_ptr<float>(),tables.size()-1,cw,view_weight.data_ptr<float>(),schema==2||schema==4,tensor_weights[7].data_ptr<float>());
  }
''' + anchor)
    text = text.replace('if(compensation_mode>=3){', 'if(compensation_mode>=3&&compensation_mode<=6){')
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('compensator_mode<=6', 'compensator_mode<=9')
    text = text.replace('mycartgs.student-schedule.shared.v1', 'mycartgs.student-preparation.shared.v1')
    anchor = 'class TensorDeploymentPipeline {'
    text = text.replace(anchor, '''void preparation_camera_into(const double*,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

''' + anchor + '''
  torch::Tensor pose_host_,pose_device_;
  std::vector<torch::Tensor> pose_parts_;
  cudaStream_t pose_stream_=nullptr;
''')
    anchor = '  torch::Tensor debug_control() const'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''  pybind11::tuple render_pose_reused(pybind11::capsule handle,const torch::Tensor& bg,
      pybind11::array_t<double,pybind11::array::c_style|pybind11::array::forcecast> pose,
      const torch::Tensor& projection,const torch::Tensor& device_reference,
      int64_t width,int64_t height,int64_t budget,double ratio,bool use_ratio,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    TORCH_CHECK(pose.ndim()==2&&pose.shape(0)==4&&pose.shape(1)==4,"Current CPU camera pose must be 4x4");
    TORCH_CHECK(!pending_requested_&&!pending_inflight_,"Finish pending frame before reusing camera staging");
    c10::cuda::CUDAGuard guard(state_[0].device());
    auto stream=c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(!pose_host_.defined()||pose_stream_==stream.stream(),"Use one reused camera pipeline per stream");
    if(!pose_host_.defined()){
      pose_host_=torch::empty({35},torch::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true));
      pose_device_=torch::empty({35},state_[0].options());pose_stream_=stream.stream();
      pose_parts_={pose_device_.slice(0,0,16).view({4,4}),pose_device_.slice(0,16,32).view({4,4}),pose_device_.slice(0,32,35)};
    }
    {py::gil_scoped_release release;preparation_camera_into(pose.data(),projection,pose_host_,pose_device_);}
    return render_pending_current(handle,bg,pose_parts_[0],pose_parts_[1],pose_parts_[2],width,height,.01,
        budget,ratio,use_ratio,true,true,rest,tanx,tany);
  }
''' + anchor)
    text = text.replace('.def("_render_pose_current",&TensorDeploymentPipeline::render_pose_current)',
                        '.def("_render_pose_current",&TensorDeploymentPipeline::render_pose_current)\n    .def("_render_pose_reused",&TensorDeploymentPipeline::render_pose_reused)')
    path.write_text(text)
    path = root / 'preparation_camera.cpp'
    text = path.read_text().replace('#include <c10/cuda/CUDAGuard.h>', '#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAStream.h>\n#include <c10/cuda/CUDAException.h>')
    original = text[text.index('torch::Tensor preparation_camera('):text.index('std::vector<torch::Tensor> preparation_camera_parts')]
    start = original.index('  const double* input=')
    end = original.index('  auto output=torch::empty')
    body = original[start:end]
    body = body.replace('  const double* input=pose.data_ptr<double>();', '  ')
    body = body.replace('  c10::cuda::CUDAGuard guard(device_reference.device());\n', '')
    body = body.replace('  auto staging=torch::empty({35},torch::TensorOptions().device(at::kCPU).dtype(at::kFloat).pinned_memory(true));\n', '')
    text += '''
// 前一帧finish_pending已同步，稳定pinned缓冲可安全写入本帧。
void preparation_camera_into(const double* input,const torch::Tensor& projection,
    const torch::Tensor& staging,const torch::Tensor& output){
  TORCH_CHECK(!torch::GradMode::is_enabled()&&projection.device().is_cpu()&&projection.scalar_type()==at::kFloat&&
      projection.is_contiguous()&&projection.sizes()==at::IntArrayRef({4,4}),"Native reused camera projection");
''' + body + '''  C10_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<float>(),staging.data_ptr<float>(),35*sizeof(float),cudaMemcpyHostToDevice,c10::cuda::getCurrentCUDAStream()));
}
'''
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('StudentScheduleRasterizer', 'StudentPreparationRasterizer')
    text = text.replace('mycartgs.student-schedule.shared.v1', 'mycartgs.student-preparation.shared.v1')
    path.write_text(text)
    print('STUDENT_PREPARATION_SOURCE_READY')


if __name__ == '__main__':
    main()
