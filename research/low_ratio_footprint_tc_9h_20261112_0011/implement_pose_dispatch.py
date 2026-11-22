"""独立v3：当前CPU位姿直接进入C++调用，复用同一相机数值实现。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'joint_source_pose';shutil.copytree(HERE/'joint_source_v2',root)
    (HERE/'joint_build_pose').mkdir()
    path=root/'raster/graph_main.cu';s=path.read_text().replace('NativePendingNineRasterizer','PosePendingNineRasterizer')
    s=s.replace('mycartgs.pending-raster.shared.v2','mycartgs.pending-raster.shared.pose.v3');path.write_text(s)
    path=root/'deployment_prepare.cpp';s=path.read_text().replace('#include <torch/extension.h>',
        '#include <torch/extension.h>\n#include <pybind11/numpy.h>')
    s=s.replace('mycartgs.pending-raster.shared.v2','mycartgs.pending-raster.shared.pose.v3')
    declaration='''std::vector<torch::Tensor> preparation_camera_parts(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
'''
    s=s.replace('class TensorDeploymentPipeline {',declaration+'\nclass TensorDeploymentPipeline {')
    anchor='  torch::Tensor debug_control() const'
    method='''  pybind11::tuple render_pose_current(pybind11::capsule handle,const torch::Tensor& bg,
      pybind11::array_t<double,pybind11::array::c_style|pybind11::array::forcecast> pose,
      const torch::Tensor& projection,const torch::Tensor& device_reference,
      int64_t width,int64_t height,int64_t budget,double ratio,bool use_ratio,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    TORCH_CHECK(pose.ndim()==2&&pose.shape(0)==4&&pose.shape(1)==4,"Current CPU camera pose must be 4x4");
    auto input=torch::from_blob(const_cast<double*>(pose.data()),{4,4},torch::TensorOptions().dtype(at::kDouble));
    std::vector<torch::Tensor> camera;
    {
      py::gil_scoped_release release;
      camera=preparation_camera_parts(input,projection,device_reference);
    }
    return render_pending_current(handle,bg,camera[0],camera[1],camera[2],width,height,.01,
        budget,ratio,use_ratio,true,true,rest,tanx,tany);
  }
'''
    assert anchor in s;s=s.replace(anchor,method+anchor)
    anchor='    .def("_render_pending_current",&TensorDeploymentPipeline::render_pending_current)'
    assert anchor in s;s=s.replace(anchor,anchor+'\n    .def("_render_pose_current",&TensorDeploymentPipeline::render_pose_current)')
    path.write_text(s)
    print('CURRENT_POSE_NATIVE_SOURCE_READY')


if __name__=='__main__':main()
