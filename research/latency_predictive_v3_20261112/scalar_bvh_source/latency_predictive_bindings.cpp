#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>
#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <vector>
#include "v2_latency_pid_core.h"
#include "latency_view_summary.h"

namespace py=pybind11;
namespace {
latency_pid::Config configuration(const py::dict& options) {
  latency_pid::Config c;c.target_ms=.2;c.max_increase=.35;c.max_decrease=.4;
  py::dict remaining=options.attr("copy")();
#define OPTION(name) if(remaining.contains(#name)){c.name=py::cast<decltype(c.name)>(remaining[#name]);remaining.attr("pop") (#name);}
  OPTION(target_ms) OPTION(max_increase) OPTION(max_decrease)
  OPTION(cut_translation) OPTION(cut_rotation_degrees)
  OPTION(initial_budget) OPTION(min_budget) OPTION(max_budget)
#undef OPTION
  if(py::len(remaining))throw std::invalid_argument("Unknown PID option: "+py::cast<std::string>(py::str(remaining)));
  c.validate();return c;
}

class AsyncPredictiveV3 {
  struct Slot {
    cudaEvent_t begin=nullptr,end=nullptr;
    int64_t frame=0,epoch=0,k=0,available=0;
    double time=0.;
    bool valid=false,external=false;
    latency_predictive::View view;
  };
  latency_pid::Config config_;
  latency_predictive::Core core_;
  latency_predictive::ViewSummary summary_;
  latency_predictive::View current_view_;
  int64_t last_requested_=3000;
  int device_,recording_=-1;
  cudaStream_t stream_=nullptr;
  bool bound_stream_=false,closed_=false,has_pose_=false;
  std::vector<Slot> slots_;
  std::deque<int> pending_,free_;
  std::array<double,16> pose_{};
  int64_t epoch_=0,frame_=0,stale_=0,dropped_=0,resets_=0,last_feedback_frame_=-1;
  double request_time_=0.,last_request_time_=-1.,last_ms_=0.,last_view_us_=0.;
  int64_t last_sample_k_=0;
  int64_t feedback_samples_=0,superseded_=0;

  void check_stream() {
    TORCH_CHECK(!closed_,"PID controller is closed");
    cudaStream_t current=c10::cuda::getCurrentCUDAStream(device_);
    if(!bound_stream_){stream_=current;bound_stream_=true;}
    TORCH_CHECK(current==stream_,"One PID instance requires serial calls on one CUDA stream");
  }
  void reset() {core_.reset();++epoch_;++resets_;}
  void poll_native() {
    Slot latest;bool ready=false;float latest_ms=0.;
    while(!pending_.empty()) {
      int index=pending_.front();Slot& s=slots_[index];
      const auto status=cudaEventQuery(s.end);
      if(status==cudaErrorNotReady)break;
      C10_CUDA_CHECK(status);
      if(s.valid && s.epoch==epoch_) {
        float ms=0.;C10_CUDA_CHECK(cudaEventElapsedTime(&ms,s.begin,s.end));
        if(ready)++superseded_;
        latest=s;latest_ms=ms;ready=true;++feedback_samples_;
      } else ++stale_;
      pending_.pop_front();free_.push_back(index);
    }
    // 积压反馈只用最新已完成样本控制，避免在同一当前K上连续应用旧误差。
    if(ready){core_.observe(latest_ms,latest.time,latest.k,latest.available,latest.view);
      last_feedback_frame_=latest.frame;last_ms_=latest_ms;last_sample_k_=latest.k;}
  }
public:
  AsyncPredictiveV3(const py::dict& options,int device,int count):config_(configuration(options)),core_(config_.target_ms,config_.initial_budget,config_.min_budget,config_.max_budget),device_(device) {
    core_.max_increase=config_.max_increase;core_.max_decrease=config_.max_decrease;
    TORCH_CHECK(count>=1 && count<=64,"PID Event ring must have 1..64 slots");
    c10::cuda::CUDAGuard guard(device_);
    slots_.resize(count);
    try {
      for(int i=0;i<count;++i){
        C10_CUDA_CHECK(cudaEventCreate(&slots_[i].begin));
        C10_CUDA_CHECK(cudaEventCreate(&slots_[i].end));free_.push_back(i);
      }
    } catch(...) {close();throw;}
  }
  ~AsyncPredictiveV3(){close();}
  void close() noexcept {
    if(closed_)return;
    int previous=0;cudaGetDevice(&previous);cudaSetDevice(device_);
    for(auto& s:slots_){if(s.begin)cudaEventDestroy(s.begin);if(s.end)cudaEventDestroy(s.end);}
    cudaSetDevice(previous);closed_=true;
  }
  int64_t request(const std::vector<double>& pose,bool force_reset,double time) {
    c10::cuda::CUDAGuard guard(device_);check_stream();
    TORCH_CHECK(recording_<0,"End the previous PID render before starting another request");
    TORCH_CHECK(pose.empty() || pose.size()==16,"Pose must be a host camera-to-world 4x4 matrix");
    for(double x:pose)TORCH_CHECK(std::isfinite(x),"Nonfinite PID pose");
    if(time<0)time=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    TORCH_CHECK(std::isfinite(time) && time>last_request_time_,"PID request time must increase");
    if(!pose.empty() && has_pose_) {
      double distance=0.,trace=0.;
      for(int i=0;i<3;++i){const double v=pose[i*4+3]-pose_[i*4+3];distance+=v*v;
        for(int j=0;j<3;++j)trace+=pose[i*4+j]*pose_[i*4+j];}
      const double cosine=std::max(-1.,std::min(1.,(trace-1.)*.5));
      force_reset=force_reset || distance>config_.cut_translation*config_.cut_translation ||
          cosine<std::cos(config_.cut_rotation_degrees*3.14159265358979323846/180.);
    }
    if(force_reset)reset();
    if(!pose.empty()){std::copy(pose.begin(),pose.end(),pose_.begin());has_pose_=true;}
    poll_native();request_time_=time;last_request_time_=time;++frame_;
    const auto view_start=std::chrono::steady_clock::now();current_view_=summary_.evaluate(pose);
    last_view_us_=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-view_start).count();
    last_requested_=core_.budget(current_view_);return last_requested_;
  }
  int begin(int64_t actual,int64_t available,bool external) {
    c10::cuda::CUDAGuard guard(device_);check_stream();
    TORCH_CHECK(frame_>0 && recording_<0 && actual>=0 && available>=actual,"Invalid PID render receipt");
    core_.applied(actual);
    if(free_.empty()){++dropped_;return -1;}
    const int index=free_.front();free_.pop_front();auto& s=slots_[index];
    s.frame=frame_;s.epoch=epoch_;s.time=request_time_;s.k=actual;s.available=available;s.valid=true;s.external=external;s.view=current_view_;
    if(!external)C10_CUDA_CHECK(cudaEventRecord(s.begin,stream_));recording_=index;return index;
  }
  int start(int64_t actual,int64_t available){return begin(actual,available,false);}
  std::array<int64_t,3> start_native(int64_t actual,int64_t available){
    const int index=begin(actual,available,true);
    if(index<0)return {-1,0,0};
    return {index,reinterpret_cast<int64_t>(slots_[index].begin),reinterpret_cast<int64_t>(slots_[index].end)};
  }
  void finish(int index,bool valid) {
    if(index<0)return;
    c10::cuda::CUDAGuard guard(device_);check_stream();
    TORCH_CHECK(recording_==index,"PID timing token does not match active render");
    slots_[index].valid=valid;
    if(!slots_[index].external || !valid)C10_CUDA_CHECK(cudaEventRecord(slots_[index].end,stream_));
    pending_.push_back(index);recording_=-1;
  }
  void configure_view(py::array_t<float,py::array::c_style|py::array::forcecast> data,
      py::array_t<float,py::array::c_style|py::array::forcecast> projection,int width,int height){
    TORCH_CHECK(!summary_.configured && !closed_,"Configure one immutable map per predictive controller");
    TORCH_CHECK(data.ndim()==2&&data.shape(1)==6&&projection.size()==16,"Predictive summary layout");
    auto a=data.unchecked<2>();std::vector<latency_predictive::Anchor> anchors;
    for(int64_t i=0;i<data.shape(0);++i){for(int j=0;j<6;++j)TORCH_CHECK(std::isfinite(a(i,j)),"Nonfinite Anchor summary");
      TORCH_CHECK(a(i,3)>0&&a(i,4)>=0&&a(i,5)>=0,"Invalid Anchor summary");
      anchors.push_back({a(i,0),a(i,1),a(i,2),a(i,3),a(i,4),a(i,5)});}
    std::array<float,16> pm;for(int j=0;j<16;++j){pm[j]=projection.data()[j];TORCH_CHECK(std::isfinite(pm[j]),"Nonfinite projection");}
    summary_.configure(std::move(anchors),pm,width,height);
  }
  void poll(){c10::cuda::CUDAGuard guard(device_);check_stream();poll_native();}
  void target(double value){
    TORCH_CHECK(!closed_,"PID controller is closed");
    TORCH_CHECK(recording_<0,"Cannot change PID target inside a timed render");
    core_.target(value);++epoch_;++resets_;
  }
  py::dict stats() const {
    py::dict value;
    value["target_ms"]=core_.target_ms;value["requested_budget"]=last_requested_;
    value["effective_budget"]=core_.applied_k;value["filtered_ms"]=core_.filtered_ms;
    value["integral"]=core_.integral;value["derivative"]=core_.derivative;value["updates"]=core_.updates;
    value["last_feedback_frame"]=last_feedback_frame_;value["last_feedback_K"]=last_sample_k_;
    value["last_feedback_ms"]=last_ms_;value["epoch"]=epoch_;value["requests"]=frame_;
    value["pending_samples"]=pending_.size();value["dropped_samples"]=dropped_;
    value["stale_samples"]=stale_;value["resets"]=resets_;value["event_slots"]=slots_.size();
    value["predicted_ms"]=core_.predicted_ms;value["predicted_available"]=core_.predicted_available;
    value["footprint"]=core_.current_footprint;value["predicted_pool_limited"]=core_.pool_limited;
    value["view_summary_us"]=last_view_us_;value["model_resets"]=core_.model_resets;
    value["summary_host_bytes"]=summary_.bytes();
    value["feedback_samples"]=feedback_samples_;value["superseded_samples"]=superseded_;
    return value;
  }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  py::class_<AsyncPredictiveV3>(module,"AsyncPID")
    .def(py::init<const py::dict&,int,int>(),py::arg("config"),py::arg("device"),py::arg("event_slots")=8)
    .def("request",&AsyncPredictiveV3::request,py::arg("pose")=std::vector<double>(),py::arg("reset")=false,py::arg("timestamp_s")=-1.)
    .def("start",&AsyncPredictiveV3::start).def("finish",&AsyncPredictiveV3::finish,py::arg("token"),py::arg("valid")=true)
    .def("start_native",&AsyncPredictiveV3::start_native)
    .def("configure_view",&AsyncPredictiveV3::configure_view)
    .def("poll",&AsyncPredictiveV3::poll).def("set_target",&AsyncPredictiveV3::target)
    .def("stats",&AsyncPredictiveV3::stats).def("close",&AsyncPredictiveV3::close);
}
