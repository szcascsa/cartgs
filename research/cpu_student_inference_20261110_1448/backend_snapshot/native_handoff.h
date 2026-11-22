// 仅冻结推理：C++相机打包/属性视图/三槽位上传调度；网络在CPU执行。
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>
namespace py=pybind11;
static void cuda_ok(cudaError_t error){TORCH_CHECK(error==cudaSuccess,"Native handoff CUDA: ",cudaGetErrorString(error));}
static py::dict stats_dict(const std::vector<double>& s){
  py::dict d;const char* names[]={"total_gaussians","anchors","visible_anchors","candidate_gaussians","eligible_candidates","requested_budget","active_gaussians","ratio","workspace_bytes","threads"};
  for(int i=0;i<10;++i)d[names[i]]=s[i];return d;
}
static py::dict packet_dict(const std::vector<T>& v,int degree){
  py::dict inputs;const char* names[]={"means3D","opacities","scales","rotations","dc","shs","means2D"};
  for(int i=0;i<7;++i)inputs[names[i]]=v[i];inputs["colors_precomp"]=py::none();inputs["cov3D_precomp"]=py::none();
  py::dict packet;packet["sh_degree"]=degree;packet["inputs"]=inputs;return packet;
}
static py::tuple prepare_cpu(std::shared_ptr<CpuPipelineMid> pipeline,py::object camera,int kind,double ratio,int budget,bool local,bool compensation,bool timing,std::optional<T> output,std::optional<T> ids,int degree){
  T vm=camera.attr("world_view_transform").cast<T>().contiguous(),pm=camera.attr("full_proj_transform").cast<T>().contiguous(),center=camera.attr("camera_center").cast<T>().contiguous();
  int width=camera.attr("image_width").cast<int>(),height=camera.attr("image_height").cast<int>();float near=camera.attr("znear").cast<float>();
  std::tuple<T,T,std::vector<double>,std::vector<double>> result;
  {py::gil_scoped_release release;result=pipeline->prepare(vm,pm,center,width,height,near,kind,ratio,budget,local,compensation,timing,output,ids);}
  auto selected=std::get<1>(result);auto stats=stats_dict(std::get<2>(result));if(timing)stats["stage_ms"]=std::get<3>(result);
  return py::make_tuple(packet_dict(pipeline->packet_views(std::get<0>(result),selected.numel()),degree),selected,stats);
}
class NativeHandoffMid;
struct NativeTicketMid {
  std::shared_ptr<NativeHandoffMid> owner;int slot;uint64_t generation;py::dict packet,stats;py::object camera;T ids;bool released=false;
  py::dict wait();void finish();
};
class NativeHandoffMid:public std::enable_shared_from_this<NativeHandoffMid>{
  struct Slot {T host,gpu,ids;cudaEvent_t ready=nullptr,done=nullptr;uint64_t generation=0;bool busy=false,has_done=false;};
  std::shared_ptr<CpuPipelineMid> pipeline_;std::vector<Slot> slots_;cudaStream_t copy_=nullptr;
  int device_,degree_,cursor_=0;int64_t capacity_;std::atomic<bool> submitting_{false};py::object namespace_type_;
public:
  NativeHandoffMid(std::shared_ptr<CpuPipelineMid> pipeline,int degree,int slots=3,int device=0)
      :pipeline_(pipeline),device_(device),degree_(degree),capacity_(pipeline->total()*pipeline->width()),namespace_type_(py::module_::import("types").attr("SimpleNamespace")){
    TORCH_CHECK(slots==2||slots==3,"Use two or three bounded handoff slots");
    c10::cuda::CUDAGuard guard(device_);cuda_ok(cudaStreamCreateWithFlags(&copy_,cudaStreamNonBlocking));slots_.resize(slots);
    try{for(auto& s:slots_){
      s.host=at::empty({capacity_+51},at::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true));
      s.ids=at::empty({pipeline_->total()},at::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
      s.gpu=at::empty({capacity_+51},at::TensorOptions().dtype(at::kFloat).device(at::kCUDA,device_));
      cuda_ok(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));cuda_ok(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming));
    }}catch(...){release_cuda();throw;}
  }
  ~NativeHandoffMid(){release_cuda();}
  void release_cuda()noexcept{
    int old=0;cudaGetDevice(&old);cudaSetDevice(device_);
    if(copy_)cudaStreamSynchronize(copy_);
    for(auto& s:slots_){if(s.has_done&&s.done)cudaEventSynchronize(s.done);if(s.ready)cudaEventDestroy(s.ready);if(s.done)cudaEventDestroy(s.done);s.ready=nullptr;s.done=nullptr;}
    if(copy_)cudaStreamDestroy(copy_);copy_=nullptr;cudaSetDevice(old);
  }
  std::shared_ptr<NativeTicketMid> prepare(py::object camera,std::optional<int64_t> budget,std::optional<double> ratio_budget,std::optional<double> global_ratio,bool local=true,bool compensation=true){
    TORCH_CHECK(int(budget.has_value())+int(ratio_budget.has_value())+int(global_ratio.has_value())==1,"Use one CPU budget request");
    TORCH_CHECK(!submitting_.exchange(true),"Native handoff requires one submitting thread");
    struct Reset{std::atomic<bool>& value;~Reset(){value=false;}}reset{submitting_};
    auto& s=slots_[cursor_];TORCH_CHECK(!s.busy,"Consume/finish previous ticket before reusing its slot");
    T vm=camera.attr("world_view_transform").cast<T>().contiguous(),pm=camera.attr("full_proj_transform").cast<T>().contiguous();
    T projection=camera.attr("projection_matrix").cast<T>().contiguous(),center=camera.attr("camera_center").cast<T>().contiguous();
    TORCH_CHECK(projection.device().is_cpu()&&projection.scalar_type()==at::kFloat&&projection.numel()==16,"CPU projection matrix");
    for(int j=0;j<16;++j)TORCH_CHECK(std::isfinite(projection.data_ptr<float>()[j]),"Nonfinite projection matrix");
    int width=camera.attr("image_width").cast<int>(),height=camera.attr("image_height").cast<int>();float near=camera.attr("znear").cast<float>();
    int kind=budget?1:global_ratio?2:0;double ratio=global_ratio.value_or(ratio_budget.value_or(0));
    TORCH_CHECK(!budget||(*budget>=0&&*budget<=std::numeric_limits<int>::max()),"Invalid budget");
    std::tuple<T,T,std::vector<double>,std::vector<double>> result;
    int64_t n;int index=cursor_;
    {
      py::gil_scoped_release release;c10::cuda::CUDAGuard guard(device_);
      if(s.has_done)cuda_ok(cudaEventSynchronize(s.done));
      result=pipeline_->prepare(vm,pm,center,width,height,near,kind,ratio,int(budget.value_or(0)),local,compensation,false,s.host.narrow(0,0,capacity_),s.ids);
      n=std::get<0>(result).numel();float* dst=s.host.data_ptr<float>()+n;
      std::memcpy(dst,vm.data_ptr<float>(),16*sizeof(float));std::memcpy(dst+16,projection.data_ptr<float>(),16*sizeof(float));
      std::memcpy(dst+32,pm.data_ptr<float>(),16*sizeof(float));std::memcpy(dst+48,center.data_ptr<float>(),3*sizeof(float));
      cuda_ok(cudaMemcpyAsync(s.gpu.data_ptr<float>(),s.host.data_ptr<float>(),(n+51)*sizeof(float),cudaMemcpyHostToDevice,copy_));cuda_ok(cudaEventRecord(s.ready,copy_));
      ++s.generation;s.busy=true;cursor_=(cursor_+1)%slots_.size();
    }
    auto ticket=std::make_shared<NativeTicketMid>();ticket->owner=shared_from_this();ticket->slot=index;ticket->generation=s.generation;
    ticket->ids=std::get<1>(result);ticket->stats=stats_dict(std::get<2>(result));
    ticket->packet=packet_dict(pipeline_->packet_views(s.gpu.narrow(0,0,n),ticket->ids.numel()),degree_);
    ticket->camera=namespace_type_();
    for(const char* name:{"image_width","image_height","FoVx","FoVy","znear","zfar"})ticket->camera.attr(name)=camera.attr(name);
    int64_t offset=n;
    for(const char* name:{"world_view_transform","projection_matrix","full_proj_transform"}){ticket->camera.attr(name)=s.gpu.narrow(0,offset,16).view({4,4});offset+=16;}
    ticket->camera.attr("camera_center")=s.gpu.narrow(0,offset,3);return ticket;
  }
  void wait(int slot,uint64_t generation){
    auto& s=slots_.at(slot);TORCH_CHECK(s.generation==generation&&s.busy,"Native ticket generation mismatch");
    c10::cuda::CUDAGuard guard(device_);cuda_ok(cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device_).stream(),s.ready,0));
  }
  void finish(int slot,uint64_t generation){
    auto& s=slots_.at(slot);TORCH_CHECK(s.generation==generation&&s.busy,"Native ticket generation mismatch");
    c10::cuda::CUDAGuard guard(device_);auto stream=c10::cuda::getCurrentCUDAStream(device_).stream();
    cuda_ok(cudaStreamWaitEvent(stream,s.ready,0));cuda_ok(cudaEventRecord(s.done,stream));s.has_done=true;s.busy=false;
  }
  void synchronize(){py::gil_scoped_release release;c10::cuda::CUDAGuard guard(device_);cuda_ok(cudaStreamSynchronize(copy_));for(auto& s:slots_)if(s.has_done)cuda_ok(cudaEventSynchronize(s.done));}
  py::dict bytes()const{py::dict d;d["host"]=slots_.size()*((capacity_+51)*4+pipeline_->total()*8);d["device"]=slots_.size()*(capacity_+51)*4;d["slots"]=slots_.size();return d;}
};
py::dict NativeTicketMid::wait(){TORCH_CHECK(!released,"CPU preparation ticket already released");owner->wait(slot,generation);return packet;}
void NativeTicketMid::finish(){TORCH_CHECK(!released,"CPU preparation ticket released twice");owner->finish(slot,generation);released=true;}
static void bind_handoff(py::module_& m){
  py::class_<NativeTicketMid,std::shared_ptr<NativeTicketMid>>(m,"PreparedTicket")
    .def("wait",&NativeTicketMid::wait).def("finish",&NativeTicketMid::finish)
    .def_readonly("packet",&NativeTicketMid::packet).def_readonly("camera",&NativeTicketMid::camera).def_readonly("ids",&NativeTicketMid::ids)
    .def_readonly("stats",&NativeTicketMid::stats).def_readonly("released",&NativeTicketMid::released);
  py::class_<NativeHandoffMid,std::shared_ptr<NativeHandoffMid>>(m,"NativeHandoff")
    .def(py::init<std::shared_ptr<CpuPipelineMid>,int,int,int>(),py::arg("pipeline"),py::arg("sh_degree"),py::arg("slots")=3,py::arg("device")=0)
    .def("prepare",&NativeHandoffMid::prepare,py::arg("camera"),py::arg("budget")=py::none(),py::arg("ratio_budget")=py::none(),py::arg("global_ratio")=py::none(),py::arg("view_conditioned")=true,py::arg("compensation")=true)
    .def("synchronize",&NativeHandoffMid::synchronize).def("bytes",&NativeHandoffMid::bytes);
}
