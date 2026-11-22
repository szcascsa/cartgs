"""精确候选：控制帧DtoH在依赖准备完成的辅助Stream上与Rasterizer重叠。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_async_control'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_async_control').mkdir()
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    anchor = '  bool small_radix_finish_=false;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
  bool overlap_control_=false,control_queued_=false;
  cudaStream_t control_stream_=nullptr;
  cudaEvent_t control_ready_=nullptr,control_copied_=nullptr;
''')
    anchor = '    return {pending_attributes_,pending_ids_,pending_frame_};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    if(overlap_control_){
      c10::cuda::CUDAGuard guard(state_[0].device());
      if(!control_ready_){
        control_stream_=c10::cuda::getStreamFromPool(false,state_[0].get_device()).stream();
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&control_ready_,cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&control_copied_,cudaEventDisableTiming));
      }
      C10_CUDA_CHECK(cudaEventRecord(control_ready_,pending_stream_));
      C10_CUDA_CHECK(cudaStreamWaitEvent(control_stream_,control_ready_,0));
      C10_CUDA_CHECK(cudaMemcpyAsync(host_control_.data_ptr(),pending_frame_.data_ptr(),pending_frame_.nbytes(),cudaMemcpyDeviceToHost,control_stream_));
      C10_CUDA_CHECK(cudaEventRecord(control_copied_,control_stream_));
      control_queued_=true;
    }
''' + anchor)
    anchor = '    {py::gil_scoped_release release;readback(pending_frame_,host_control_);}'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    {py::gil_scoped_release release;
      if(control_queued_){
        // 仍等待本帧Rasterizer完成，不能将返回后的GPU工作隐藏到计时外。
        C10_CUDA_CHECK(cudaStreamSynchronize(pending_stream_));
        C10_CUDA_CHECK(cudaEventSynchronize(control_copied_));control_queued_=false;
      }else readback(pending_frame_,host_control_);
    }''')
    anchor = '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
  void overlap_control(bool enabled){
    TORCH_CHECK(!pending_inflight_&&!pending_requested_,"Finish pending frame before changing control stream");
    overlap_control_=enabled;
  }
  ~TensorDeploymentPipeline(){
    // 只等本实例的辅助复制；确保Host pinned缓冲和设备Frame在析构前仍有效。
    if(control_queued_)cudaEventSynchronize(control_copied_);
    if(control_ready_)cudaEventDestroy(control_ready_);
    if(control_copied_)cudaEventDestroy(control_copied_);
  }
''')
    anchor = '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish);'
    assert text.count(anchor) == 1
    text = text.replace(anchor,
        '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish)\n    .def("_overlap_control",&TensorDeploymentPipeline::overlap_control);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-async-control.shared.v1').replace('StudentJointFinalRasterizer', 'StudentAsyncControlRasterizer'))
    print('ASYNC_CONTROL_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
