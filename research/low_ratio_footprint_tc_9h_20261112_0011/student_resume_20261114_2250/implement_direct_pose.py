"""精确候选：本帧相机与请求合并为一次有界HtoD，省去重复设备端相机搬运。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_direct_pose'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_direct_pose').mkdir()
    path = root / 'preparation_camera.cpp'
    text = path.read_text()
    function = text[text.index('void preparation_camera_into('):]
    function = function.replace('preparation_camera_into', 'preparation_camera_pack')
    copy = '  C10_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr<float>(),staging.data_ptr<float>(),35*sizeof(float),cudaMemcpyHostToDevice,c10::cuda::getCurrentCUDAStream()));'
    assert function.count(copy) == 1
    path.write_text(text + '\n' + function.replace(copy, ''))
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('#include <climits>', '#include <climits>\n#include <cstring>')
    declaration = 'void preparation_camera_into(const double*,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);'
    assert text.count(declaration) == 1
    text = text.replace(declaration, declaration + '\nvoid preparation_camera_pack(const double*,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);')
    text = text.replace('  bool small_radix_finish_=false;', '''  bool small_radix_finish_=false;
  bool direct_pose_=false,direct_pose_active_=false;
  torch::Tensor request_host_,pending_camera_;
''')
    anchor = '    torch::Tensor camera,request,planes,distance,visible;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '    torch::Tensor camera,request,planes,distance,visible,packet;')
    anchor = '  void graph_work(GraphPlan& plan) {'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''  void upload_graph_request(GraphPlan& plan,const torch::Tensor& vm,const torch::Tensor& pm,
      const torch::Tensor& center,int64_t budget,double ratio,double aspect,double near,bool use_ratio,
      const torch::Tensor& attributes,const torch::Tensor& selected){
    if(!direct_pose_active_){
      deployment_upload_request(vm,pm,center,budget,ratio,aspect,near,use_ratio,plan.camera,plan.request,attributes,selected);
      return;
    }
    if(!request_host_.defined())request_host_=torch::empty({144+int64_t(sizeof(DeploymentRequest))},torch::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
    std::memcpy(request_host_.data_ptr(),pose_host_.data_ptr(),140);
    DeploymentRequest request{};request.budget=budget;request.ratio=ratio;request.aspect=aspect;request.near=near;
    request.use_ratio=use_ratio;request.attributes=attributes.data_ptr<float>();request.selected=selected.data_ptr<int64_t>();
    std::memcpy(request_host_.data_ptr<uint8_t>()+144,&request,sizeof(request));
    C10_CUDA_CHECK(cudaMemcpyAsync(plan.packet.data_ptr(),request_host_.data_ptr(),request_host_.nbytes(),cudaMemcpyHostToDevice,c10::cuda::getCurrentCUDAStream()));
  }

''' + anchor)
    anchor = '''      created->camera=torch::empty({35},state_[0].options());
      created->request=torch::empty({sizeof(DeploymentRequest)},state_[0].options().dtype(at::kByte));'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''      created->packet=torch::empty({144+int64_t(sizeof(DeploymentRequest))},state_[0].options().dtype(at::kByte));
      created->camera=created->packet.slice(0,0,140).view(at::kFloat);
      created->request=created->packet.slice(0,144,144+sizeof(DeploymentRequest));''')
    anchor = '''        deployment_upload_request(vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,
            created->camera,created->request,attributes,selected);'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        upload_graph_request(*created,vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,attributes,selected);')
    anchor = '''    deployment_upload_request(vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,
        plan->camera,plan->request,attributes,selected);'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '    upload_graph_request(*plan,vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,attributes,selected);')
    anchor = '      pending_attributes_=attributes;pending_ids_=selected;pending_frame_=plan->buffers[0];pending_capacity_=capacity;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + 'pending_camera_=plan->camera;')
    anchor = '''      output=render_pending_shared(raster,bg,pending[0],pending[2],pending[1].numel(),rest,
          vm,pm,tanx,tany,height,width,sh_degree_,camera);'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''      if(direct_pose_active_){
        TORCH_CHECK(pending_camera_.defined(),"Direct pose requires GPU graph preparation");
        auto view=pending_camera_.slice(0,0,16).view({4,4}),projection=pending_camera_.slice(0,16,32).view({4,4});
        output=render_pending_shared(raster,bg,pending[0],pending[2],pending[1].numel(),rest,
            view,projection,tanx,tany,height,width,sh_degree_,pending_camera_.slice(0,32,35));
      }else output=render_pending_shared(raster,bg,pending[0],pending[2],pending[1].numel(),rest,
          vm,pm,tanx,tany,height,width,sh_degree_,camera);''')
    anchor = '''    {py::gil_scoped_release release;preparation_camera_into(pose.data(),projection,pose_host_,pose_device_);}
    return render_pending_current(handle,bg,pose_parts_[0],pose_parts_[1],pose_parts_[2],width,height,.01,
        budget,ratio,use_ratio,true,true,rest,tanx,tany);'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    // 沿用原准备Graph的容量限制；未使用Graph的请求仍按原路径提交当前相机。
    const bool direct=direct_pose_&&graph_enabled_&&device_counts_&&total_&&
        (total_<=16384||shrinking_radix_)&&(use_ratio?(std::isfinite(ratio)&&ratio*double(total_)<32769.):std::min(budget,total_)<=32768);
    {py::gil_scoped_release release;
      if(direct)preparation_camera_pack(pose.data(),projection,pose_host_,pose_device_);
      else preparation_camera_into(pose.data(),projection,pose_host_,pose_device_);
    }
    direct_pose_active_=direct;
    try{
      auto result=render_pending_current(handle,bg,pose_parts_[0],pose_parts_[1],pose_parts_[2],width,height,.01,
          budget,ratio,use_ratio,true,true,rest,tanx,tany);
      direct_pose_active_=false;return result;
    }catch(...){direct_pose_active_=false;throw;}''')
    anchor = '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}'
    text = text.replace(anchor, anchor + '\n  void direct_pose(bool enabled){TORCH_CHECK(!pending_inflight_,"Finish pending frame before direct pose change");direct_pose_=enabled;}')
    anchor = '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '.def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish)\n    .def("_direct_pose",&TensorDeploymentPipeline::direct_pose);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-direct-pose.shared.v1').replace('StudentJointFinalRasterizer', 'StudentDirectPoseRasterizer'))
    print('DIRECT_POSE_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
