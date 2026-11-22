// 独立实验：有界容量调度图。所有输入、相机和输出指针每帧重新提交。
#include "graph_kernels.cuh"
#include <list>
#include <memory>
#include <tuple>

using Tensor=torch::Tensor;

struct CapturedPlan {
  int capacity,W,H,variant,list_capacity;uint64_t last_used=0;
  Tensor geometry,sort,bits,directory,lists,counts,tasks;
  cudaGraphExec_t execution=nullptr;
  ~CapturedPlan(){if(execution)cudaGraphExecDestroy(execution);}
  int64_t bytes()const {
    int64_t count=0;for(const auto& t:{geometry,sort,bits,directory,lists,counts,tasks})count+=t.storage().nbytes();return count;
  }
};

struct CapacityGraphRasterizer {
  int device=-1;cudaStream_t owner=nullptr;Tensor parameters;
  std::list<std::unique_ptr<CapturedPlan>> plans;
  uint64_t clock=0,hits=0,misses=0,evictions=0;
  static constexpr int64_t Limit=1024ll<<20;

  static int64_t estimated_bytes(int capacity,int W,int H,int list_capacity) {
    int64_t tiles=((W+15)/16)*((H+15)/16),words=(capacity+31)/32,groups=(words+31)/32;
    return int64_t(capacity)*128+8192+tiles*words*4+4+tiles*groups*4+
        tiles*list_capacity*4+tiles*4+int64_t(capacity)*((tiles+255)/256)*8;
  }

  int64_t workspace_bytes()const {
    int64_t n=parameters.defined()?parameters.storage().nbytes():0;
    for(const auto& plan:plans)n+=plan->bytes();return n;
  }

  std::unique_ptr<CapturedPlan> create(int capacity,int W,int H,int variant,int list_capacity,torch::TensorOptions options) {
    auto plan=std::make_unique<CapturedPlan>();
    plan->capacity=capacity;plan->W=W;plan->H=H;plan->variant=variant;plan->list_capacity=list_capacity;
    const int tiles=((W+15)/16)*((H+15)/16),words=(capacity+31)/32,groups=(words+31)/32;
    const int64_t bit_bytes=int64_t(tiles)*words*4+4;
    TORCH_CHECK(bit_bytes<=256ll<<20,"Graph bitset exceeds 256 MiB");
    auto buffer=[&](int64_t n){return torch::empty({n},options.dtype(at::kByte));};
    plan->geometry=buffer(int64_t(capacity)*96+4096);plan->sort=buffer(int64_t(capacity)*32+4096);
    plan->bits=buffer(bit_bytes);plan->directory=buffer(int64_t(tiles)*groups*4);
    plan->lists=buffer(int64_t(tiles)*list_capacity*4);plan->counts=buffer(int64_t(tiles)*4);
    plan->tasks=buffer(int64_t(capacity)*((tiles+255)/256)*8);
    TORCH_CHECK(plan->bytes()<=Limit,"Graph workspace exceeds 1 GiB");
    char* geom=reinterpret_cast<char*>(plan->geometry.data_ptr());
    using CudaRasterizer::obtain;
    float* depths;bool* clamped;int* radii;float2* xy;float* covariance;float4* conic;float* rgb;uint32_t* touched;
    obtain(geom,depths,capacity,128);obtain(geom,clamped,capacity*3,128);obtain(geom,radii,capacity,128);
    obtain(geom,xy,capacity,128);obtain(geom,covariance,capacity*6,128);obtain(geom,conic,capacity,128);
    obtain(geom,rgb,capacity*3,128);obtain(geom,touched,capacity,128);
    char* sorting=reinterpret_cast<char*>(plan->sort.data_ptr());uint32_t *key_in,*key_out,*id_in,*sorted;
    uint64_t *merge_in,*merge_out;
    obtain(sorting,key_in,capacity,128);obtain(sorting,key_out,capacity,128);obtain(sorting,id_in,capacity,128);obtain(sorting,sorted,capacity,128);
    obtain(sorting,merge_in,capacity,128);obtain(sorting,merge_out,capacity,128);
    auto bits=reinterpret_cast<uint32_t*>(plan->bits.data_ptr());auto task_count=bits+int64_t(tiles)*words;
    auto tasks=reinterpret_cast<uint2*>(plan->tasks.data_ptr());
    auto directory=reinterpret_cast<uint32_t*>(plan->directory.data_ptr());
    auto lists=reinterpret_cast<uint32_t*>(plan->lists.data_ptr());auto counts=reinterpret_cast<uint32_t*>(plan->counts.data_ptr());
    auto inputs=reinterpret_cast<const GraphInputs*>(parameters.data_ptr());
    int workers=at::cuda::getCurrentDeviceProperties()->multiProcessorCount*4;
    cudaStream_t capture;cudaGraph_t graph;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamBeginCapture(capture,cudaStreamCaptureModeThreadLocal));
    dim3 grid((W+15)/16,(H+15)/16);
    graph_preprocess(capacity,inputs,radii,xy,depths,covariance,rgb,conic,clamped,touched,key_in,id_in,grid,capture);
    blockDepthSort512<<<(capacity+511)/512,128,0,capture>>>(capacity,key_in,merge_in,key_out,sorted,capacity<=512);
    for(int width=512;width<capacity;width*=2) {
      mergeDepthSort<<<(capacity+255)/256,256,0,capture>>>(capacity,width,merge_in,merge_out,key_out,sorted,width*2>=capacity);
      std::swap(merge_in,merge_out);
    }
    C10_CUDA_CHECK(cudaMemsetAsync(bits,0,bit_bytes,capture));
    scheduleBits<false><<<(capacity+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,tasks,task_count);
    executeBitTasks<false><<<workers,128,0,capture>>>(tasks,task_count,sorted,xy,conic,radii,bits,words,grid);
    compactTileBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity);
    if(variant==5)renderMultiPixel<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,
        list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else renderCached<1><<<grid,dim3(16,16),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,
        list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    C10_CUDA_CHECK(cudaStreamEndCapture(capture,&graph));
    C10_CUDA_CHECK(cudaGraphInstantiate(&plan->execution,graph,nullptr,nullptr,0));
    C10_CUDA_CHECK(cudaGraphDestroy(graph));C10_CUDA_CHECK(cudaStreamDestroy(capture));
    return plan;
  }

  std::vector<Tensor> render(const Tensor& bg,const Tensor& xyz,const Tensor& colors,const Tensor& opacity,
      const Tensor& scales,const Tensor& rotation,double modifier,const Tensor& covariance,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
      const Tensor& dc,const Tensor& sh,int64_t degree,const Tensor& camera,int variant,int list_capacity,int plan_granularity) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Capacity graph is frozen inference only");
    TORCH_CHECK(xyz.is_cuda()&&xyz.dim()==2&&xyz.size(1)==3&&H>0&&W>0,"Graph inputs");
    TORCH_CHECK((variant==1||variant==5)&&list_capacity>0&&list_capacity<=256,"Graph rendering config");
    TORCH_CHECK(plan_granularity==0||plan_granularity==512,"Graph capacity granularity");
    c10::cuda::CUDAGuard guard(xyz.device());auto stream=c10::cuda::getCurrentCUDAStream();
    if(device>=0)TORCH_CHECK(device==xyz.get_device()&&owner==stream.stream(),"Use one graph renderer per device/stream");
    device=xyz.get_device();owner=stream.stream();
    for(const auto& t:{bg,xyz,colors,opacity,scales,rotation,covariance,view,proj,dc,sh,camera})
      TORCH_CHECK(t.is_cuda()&&t.device()==xyz.device()&&t.scalar_type()==at::kFloat,"Input CUDA FP32/device");
    TORCH_CHECK(bg.numel()==3&&view.dim()==2&&view.size(0)==4&&view.size(1)==4&&proj.dim()==2&&proj.size(0)==4&&proj.size(1)==4&&camera.numel()==3,"Graph camera dimensions");
    const int P=xyz.size(0);
    auto out=P?torch::empty({3,H,W},xyz.options()):torch::zeros({3,H,W},xyz.options());
    auto output_radii=P?torch::empty({P},xyz.options().dtype(at::kInt)):torch::zeros({P},xyz.options().dtype(at::kInt));
    if(!P)return {out,output_radii};
    auto x=xyz.contiguous(),s=scales.contiguous(),q=rotation.contiguous(),a=opacity.contiguous(),d=dc.contiguous(),h=sh.contiguous();
    auto color=colors.contiguous(),cov=covariance.contiguous();
    if(!parameters.defined())parameters=torch::empty({int64_t(sizeof(GraphInputs))},xyz.options().dtype(at::kByte));
    int capacity=512;
    if(plan_granularity==512)capacity=((P+511)/512)*512;
    else while(capacity<P)capacity*=2;
    CapturedPlan* selected=nullptr;
    for(auto& plan:plans)if(plan->capacity==capacity&&plan->W==W&&plan->H==H&&plan->variant==variant&&plan->list_capacity==list_capacity){selected=plan.get();break;}
    if(!selected) {
      ++misses;int64_t required=estimated_bytes(capacity,W,H,list_capacity);
      TORCH_CHECK(required+parameters.storage().nbytes()<=Limit,"Requested graph workspace exceeds 1 GiB");
      while(!plans.empty()&&(plans.size()>=8||workspace_bytes()+required>Limit)) {
        auto oldest=std::min_element(plans.begin(),plans.end(),[](const auto& l,const auto& r){return l->last_used<r->last_used;});
        plans.erase(oldest);++evictions;
      }
      auto plan=create(capacity,W,H,variant,list_capacity,xyz.options());
      TORCH_CHECK(plan->bytes()==required,"Graph workspace accounting mismatch");
      selected=plan.get();plans.push_back(std::move(plan));
    } else ++hits;
    selected->last_used=++clock;
    GraphInputs inputs{};inputs.P=P;inputs.D=degree;inputs.M=sh.numel()?sh.size(1):0;inputs.W=W;inputs.H=H;
    inputs.focal_x=W/(2.0f*float(tanx));inputs.focal_y=H/(2.0f*float(tany));inputs.tanx=tanx;inputs.tany=tany;inputs.modifier=modifier;
    inputs.xyz=x.data_ptr<float>();inputs.scales=s.data_ptr<float>();inputs.rotations=q.data_ptr<float>();inputs.opacity=a.data_ptr<float>();
    inputs.dc=d.data_ptr<float>();inputs.sh=h.data_ptr<float>();inputs.covariance=cov.numel()?cov.data_ptr<float>():nullptr;
    inputs.colors=color.numel()?color.data_ptr<float>():nullptr;
    inputs.view=view.data_ptr<float>();inputs.projection=proj.data_ptr<float>();inputs.camera=camera.data_ptr<float>();inputs.background=bg.data_ptr<float>();
    inputs.v0=view.stride(0);inputs.v1=view.stride(1);inputs.p0=proj.stride(0);inputs.p1=proj.stride(1);inputs.c0=camera.stride(0);
    inputs.image=out.data_ptr<float>();inputs.radii=output_radii.data_ptr<int>();
    updateGraphInputs<<<1,1,0,stream>>>(inputs,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));
    C10_CUDA_CHECK(cudaGraphLaunch(selected->execution,stream));C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,output_radii};
  }
  std::vector<int64_t> stats()const{return {int64_t(hits),int64_t(misses),int64_t(evictions),int64_t(plans.size()),workspace_bytes()};}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<CapacityGraphRasterizer>(m,"GraphRasterizer").def(pybind11::init<>())
      .def("render",&CapacityGraphRasterizer::render,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("workspace_bytes",&CapacityGraphRasterizer::workspace_bytes).def("stats",&CapacityGraphRasterizer::stats);
}
