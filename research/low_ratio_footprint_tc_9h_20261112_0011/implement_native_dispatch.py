"""独立v2候选：一次pybind调用串联当前帧准备、GPU渲染及原finite检查。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'joint_source_v2';shutil.copytree(HERE/'joint_source',root)
    (HERE/'joint_build_v2').mkdir()
    header='''#pragma once
#include <torch/extension.h>
std::vector<torch::Tensor> render_pending_shared(void* handle,const torch::Tensor& bg,
    const torch::Tensor& flat,const torch::Tensor& frame,int64_t capacity,int64_t rest,
    const torch::Tensor& view,const torch::Tensor& proj,double tanx,double tany,
    int64_t H,int64_t W,int64_t degree,const torch::Tensor& camera);
'''
    (root/'pending_raster_api.h').write_text(header)
    path=root/'raster/graph_main.cu';s=path.read_text().replace('PendingNineRasterizer','NativePendingNineRasterizer')
    s=s.replace('#include <list>','#include <list>\n#include "../pending_raster_api.h"')
    before='void register_pending_raster(pybind11::module_& m) {'
    exported='''std::vector<Tensor> render_pending_shared(void* handle,const Tensor& bg,
    const Tensor& flat,const Tensor& frame,int64_t capacity,int64_t rest,
    const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
    int64_t degree,const Tensor& camera){
  auto& raster=*static_cast<std::shared_ptr<NativePendingNineRasterizer>*>(handle);
  return raster->render_pending(bg,flat,frame,capacity,rest,view,proj,tanx,tany,H,W,degree,camera);
}

'''
    assert before in s;s=s.replace(before,exported+before)
    s=s.replace('pybind11::class_<NativePendingNineRasterizer>(m,','pybind11::class_<NativePendingNineRasterizer,std::shared_ptr<NativePendingNineRasterizer>>(m,')
    anchor='      .def("render_pending",&NativePendingNineRasterizer::render_pending,pybind11::call_guard<pybind11::gil_scoped_release>())'
    assert anchor in s
    s=s.replace(anchor,anchor+'''
      .def("_shared_handle",[](std::shared_ptr<NativePendingNineRasterizer> self){
        auto* handle=new std::shared_ptr<NativePendingNineRasterizer>(std::move(self));
        return pybind11::capsule(handle,"mycartgs.pending-raster.shared.v2",[](PyObject* object){
          delete static_cast<std::shared_ptr<NativePendingNineRasterizer>*>(PyCapsule_GetPointer(object,"mycartgs.pending-raster.shared.v2"));
        });
      })''')
    path.write_text(s)
    path=root/'deployment_prepare.cpp';s=path.read_text().replace('#include "deployment_device_control.h"',
        '#include "deployment_device_control.h"\n#include "pending_raster_api.h"')
    anchor='  torch::Tensor debug_control() const'
    method='''  pybind11::tuple render_pending_current(pybind11::capsule handle,const torch::Tensor& bg,
      const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,
      bool local,bool compensation,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    void* raster=handle.get_pointer();
    TORCH_CHECK(std::string(handle.name())=="mycartgs.pending-raster.shared.v2","Pending renderer handle schema");
    std::vector<torch::Tensor> output;
    {
      py::gil_scoped_release release;
      auto pending=begin_pending(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);
      output=render_pending_shared(raster,bg,pending[0],pending[2],pending[1].numel(),rest,
          vm,pm,tanx,tany,height,width,sh_degree_,camera);
    }
    auto finished=finish_pending();
    auto ids=finished[0].cast<torch::Tensor>();
    return py::make_tuple(output[0],output[1].slice(0,0,ids.numel()),ids,finished[1]);
  }
'''
    assert anchor in s;s=s.replace(anchor,method+anchor)
    anchor='    .def("_finish_pending",&TensorDeploymentPipeline::finish_pending)'
    assert anchor in s;s=s.replace(anchor,anchor+'\n    .def("_render_pending_current",&TensorDeploymentPipeline::render_pending_current)')
    path.write_text(s)
    print('NATIVE_CURRENT_FRAME_DISPATCH_SOURCE_READY')


if __name__=='__main__':main()
