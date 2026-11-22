"""精确Full入口：冻结属性直接消费、当前CPU相机进入同一C++调用，图像与返回ID独立。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_full_pose'
    shutil.copytree(HERE / 'source_joint_incremental', root)
    (HERE / 'build_full_pose').mkdir()
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('  torch::Tensor request_host_,pending_camera_;',
        '  torch::Tensor request_host_,pending_camera_,full_frame_,full_host_;')
    anchor = '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}'
    assert text.count(anchor) == 1
    method = '''  pybind11::tuple render_full_pose(pybind11::capsule handle,const torch::Tensor& bg,
      pybind11::array_t<double,pybind11::array::c_style|pybind11::array::forcecast> pose,
      const torch::Tensor& projection,const torch::Tensor& flat,const torch::Tensor& rows,
      int64_t width,int64_t height,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    TORCH_CHECK(!torch::GradMode::is_enabled()&&!pending_inflight_&&!pending_requested_,"Full pose requires completed frozen inference");
    TORCH_CHECK(pose.ndim()==2&&pose.shape(0)==4&&pose.shape(1)==4,"Current CPU camera pose must be 4x4");
    TORCH_CHECK(std::string(handle.name())=="mycartgs.student-joint-incremental.shared.v1","Pending renderer handle schema");
    TORCH_CHECK(flat.device()==state_[0].device()&&flat.is_contiguous()&&flat.scalar_type()==at::kFloat&&
        flat.dim()==1&&flat.numel()==total_*(17+rest)&&rows.device()==flat.device()&&rows.scalar_type()==at::kLong&&rows.numel()==total_,"Frozen Full attribute layout");
    c10::cuda::CUDAGuard guard(state_[0].device());auto stream=c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(!pose_host_.defined()||pose_stream_==stream.stream(),"Use one reused camera pipeline per stream");
    if(!pose_host_.defined()){
      pose_host_=torch::empty({35},torch::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true));
      pose_device_=torch::empty({35},state_[0].options());pose_stream_=stream.stream();
      pose_parts_={pose_device_.slice(0,0,16).view({4,4}),pose_device_.slice(0,16,32).view({4,4}),pose_device_.slice(0,32,35)};
    }
    std::vector<torch::Tensor> output;
    {
      py::gil_scoped_release release;
      if(!full_frame_.defined()){
        full_host_=torch::empty({sizeof(DeploymentFrame)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
        DeploymentFrame frame{};frame.n=frame.available=frame.budget=frame.k=total_;frame.ratio=1.;
        std::memcpy(full_host_.data_ptr(),&frame,sizeof(frame));
        full_frame_=torch::empty({sizeof(DeploymentFrame)},state_[0].options().dtype(at::kByte));
        full_frame_.copy_(full_host_,true);
      }
      preparation_camera_into(pose.data(),projection,pose_host_,pose_device_);
      output=render_pending_shared(handle.get_pointer(),bg,flat,full_frame_,total_,rest,pose_parts_[0],pose_parts_[1],
          tanx,tany,height,width,sh_degree_,pose_parts_[2]);
      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    py::dict stats;stats["candidate_gaussians"]=total_;stats["active_gaussians"]=total_;
    return py::make_tuple(output[0],output[1],rows.clone(),stats);
  }
'''
    text = text.replace(anchor, method + '\n' + anchor)
    anchor = '.def("_render_pose_reused",&TensorDeploymentPipeline::render_pose_reused)'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n    .def("_render_full_pose",&TensorDeploymentPipeline::render_full_pose)')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-incremental.shared.v1',
            'mycartgs.student-full-pose.shared.v1').replace('StudentJointIncrementalRasterizer', 'StudentFullPoseRasterizer'))
    print('FULL_POSE_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
