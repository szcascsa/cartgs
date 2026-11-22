// 独立实验：有界容量调度图。所有输入、相机和输出指针每帧重新提交。
#include "graph_kernels.cuh"
#include "../deployment_device_control.h"
#include "footprint_build.cuh"
#include "warp_pixels.cuh"
#include "direct_pixels.cuh"
#include "preprocess_tasks.cuh"
#include "scaling_pixels.cuh"
#include "packed_pixels.cuh"
#include "folded_pixels.cuh"
#include "approx_pixels.cuh"
#include "tensor_pixels.cuh"
#include "diagnosed_pixels.cuh"
#include "branchless_pixels.cuh"
#include "branchless_diagnosed.cuh"
#include "coherent_pixels.cuh"
#include "coherent_diagnosed.cuh"
#include "encoded_pixels.cuh"
#include "encoded_diagnosed.cuh"
#include "encoded_sign_pixels.cuh"
#include "encoded_sign_diagnosed.cuh"
#include "pending_dynamic.cuh"
#include "tile_clear.cuh"
#include <list>
#include "../pending_raster_api.h"
#include <memory>
#include <tuple>
#include <map>

using Tensor=torch::Tensor;
__global__ void updatePendingInputs(GraphInputs input,const float* flat,const DeploymentFrame* frame,int rest,GraphInputs* destination){
  const int cap=input.P;const int k=(frame->status||frame->k<0||frame->k>cap)?0:int(frame->k);
  input.P=k;input.xyz=flat;input.opacity=flat+3ll*k;input.scales=flat+4ll*k;input.rotations=flat+7ll*k;
  input.dc=flat+11ll*k;input.sh=flat+14ll*k;input.M=rest/3;
  *destination=input;
}


struct CapturedPlan {
  int capacity,W,H,variant,list_capacity;bool input_in_graph=false;uint64_t last_used=0;
  Tensor geometry,sort,bits,directory,lists,counts,tasks;
  cudaGraphExec_t execution=nullptr;
  cudaGraph_t definition=nullptr;cudaGraphNode_t input_node=nullptr;
  ~CapturedPlan(){if(execution)cudaGraphExecDestroy(execution);if(definition)cudaGraphDestroy(definition);}
  int64_t bytes()const {
    int64_t count=0;for(const auto& t:{geometry,sort,bits,directory,lists,counts,tasks})count+=t.storage().nbytes();return count;
  }
};

struct StudentTileClearRasterizer {
  int device=-1;cudaStream_t owner=nullptr;Tensor parameters;
  Tensor pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks;
  int pool_capacity=0,pool_W=0,pool_H=0;
  std::list<std::unique_ptr<CapturedPlan>> plans;
  using Key=std::tuple<int,int,int,int,int,bool>;
  std::map<Key,CapturedPlan*> lookup;
  int max_plans=64;int build_mode_=0;int pixel_mode_=0;int preprocess_threads_=256;Tensor pool_metadata;
  bool ordered_coverage_=false;
  bool device_clear_=false;
  uint64_t clock=0,hits=0,misses=0,evictions=0;
  static constexpr int64_t Limit=1024ll<<20;
  StudentTileClearRasterizer(int count=64,bool ordered=true,bool device_clear=true,int build_mode=0,int pixel_mode=0,int preprocess_threads=256):max_plans(count),build_mode_(build_mode),pixel_mode_(pixel_mode),preprocess_threads_(preprocess_threads),ordered_coverage_(ordered),device_clear_(device_clear){TORCH_CHECK(count>0&&count<=512,"Graph plan count must be in 1..512");TORCH_CHECK(build_mode>=0&&build_mode<=19&&pixel_mode>=0&&pixel_mode<=200&&(preprocess_threads==64||preprocess_threads==128||preprocess_threads==256)&&ordered,"Footprint build mode and exact coverage");}

  static int64_t estimated_bytes(int capacity,int W,int H,int list_capacity) {
    int64_t tiles=((W+15)/16)*((H+15)/16),words=(capacity+31)/32,groups=(words+31)/32;
    return int64_t(capacity)*148+8192+tiles*words*4+8+tiles*groups*4+
        tiles*list_capacity*4+tiles*4+int64_t(capacity)*((tiles+255)/256)*8;
  }

  int64_t workspace_bytes()const {
    int64_t n=parameters.defined()?parameters.storage().nbytes():0;
    if(pool_metadata.defined())n+=pool_metadata.storage().nbytes();
    for(const auto& t:{pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks})if(t.defined())n+=t.storage().nbytes();
    return n;
  }

  void reserve_pool(int capacity,int W,int H,torch::TensorOptions options) {
    if(pool_capacity>=capacity&&pool_W==W&&pool_H==H)return;
    const int tiles=((W+15)/16)*((H+15)/16),words=(capacity+31)/32,groups=(words+31)/32;
    const int64_t bit_bytes=int64_t(tiles)*words*4+8,required=estimated_bytes(capacity,W,H,256);
    TORCH_CHECK(bit_bytes<=256ll<<20&&required+sizeof(GraphInputs)<=Limit,"Shared graph pool exceeds capacity limit");
    // 同一Stream串行使用；扩容或分辨率变化立即销毁所有旧地址的计划。
    lookup.clear();plans.clear();pool_geometry=Tensor();pool_sort=Tensor();pool_bits=Tensor();pool_directory=Tensor();
    pool_lists=Tensor();pool_counts=Tensor();pool_tasks=Tensor();pool_metadata=Tensor();
    auto buffer=[&](int64_t n){return torch::empty({n},options.dtype(at::kByte));};
    pool_geometry=buffer(int64_t(capacity)*96+4096);pool_sort=buffer(int64_t(capacity)*32+4096);
    pool_bits=buffer(bit_bytes);pool_bits.zero_();pool_directory=buffer(int64_t(tiles)*groups*4);
    pool_lists=buffer(int64_t(tiles)*256*4);pool_counts=buffer(int64_t(tiles)*4);
    pool_tasks=buffer(int64_t(capacity)*((tiles+255)/256)*8);pool_metadata=buffer(int64_t(capacity)*20);
    pool_capacity=capacity;pool_W=W;pool_H=H;
    TORCH_CHECK(workspace_bytes()<=Limit,"Shared graph pool accounting exceeded limit");
  }

  void reserve(int64_t points,int W,int H,int plan_granularity,const Tensor& exemplar) {
    TORCH_CHECK(points>0&&W>0&&H>0&&(plan_granularity==0||plan_granularity==512||plan_granularity==32),"Graph reserve dimensions");
    c10::cuda::CUDAGuard guard(exemplar.device());auto stream=c10::cuda::getCurrentCUDAStream();
    if(device>=0)TORCH_CHECK(device==exemplar.get_device()&&owner==stream.stream(),"Use one graph renderer per device/stream");
    device=exemplar.get_device();owner=stream.stream();
    if(!parameters.defined())parameters=torch::empty({int64_t(sizeof(GraphInputs))},exemplar.options().dtype(at::kByte));
    int capacity=plan_granularity==32?32:512;if(plan_granularity==32)capacity=points<=512?((points+31)/32)*32:((points+511)/512)*512;else if(plan_granularity==512)capacity=((points+511)/512)*512;else while(capacity<points)capacity*=2;
    reserve_pool(capacity,W,H,exemplar.options());
  }

  std::unique_ptr<CapturedPlan> create(int capacity,int W,int H,int variant,int list_capacity,torch::TensorOptions options,bool input_in_graph) {
    auto plan=std::make_unique<CapturedPlan>();
    plan->capacity=capacity;plan->W=W;plan->H=H;plan->variant=variant;plan->list_capacity=list_capacity;plan->input_in_graph=input_in_graph;
    const int tiles=((W+15)/16)*((H+15)/16),words=(capacity+31)/32,groups=(words+31)/32;
    const int64_t bit_bytes=int64_t(tiles)*words*4+8;
    TORCH_CHECK(bit_bytes<=256ll<<20,"Graph bitset exceeds 256 MiB");
    TORCH_CHECK(pool_capacity>=capacity&&pool_W==W&&pool_H==H,"Graph pool must be reserved before capture");
    plan->geometry=pool_geometry;plan->sort=pool_sort;plan->bits=pool_bits;plan->directory=pool_directory;
    plan->lists=pool_lists;plan->counts=pool_counts;plan->tasks=pool_tasks;
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
    if(input_in_graph)updateGraphInputs<<<1,1,0,capture>>>(GraphInputs{},reinterpret_cast<GraphInputs*>(parameters.data_ptr()));
    dim3 grid((W+15)/16,(H+15)/16);
    const bool fused_tasks=build_mode_==10||build_mode_==11||build_mode_>=12;
    auto inverse=fused_tasks?reinterpret_cast<uint32_t*>(pool_metadata.data_ptr()):nullptr;
    if(build_mode_==16)clearTilePendingBits<64><<<tiles,64,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==17)clearTilePendingBits<128><<<tiles,128,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==18)clearTilePendingBits<256><<<tiles,256,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==19)clearTilePendingBits<32><<<tiles,32,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==14||build_mode_==15)clearActivePendingBits<<<std::min<int64_t>((bit_bytes/4+255)/256,workers*2),256,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_>=12)clearPendingBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,words,tiles,inputs);
    else if(fused_tasks)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);
    graph_preprocess(capacity,inputs,radii,xy,depths,covariance,rgb,conic,clamped,touched,key_in,id_in,grid,capture,
        preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr);
    if(capacity<=32)warpDepthSort32<<<1,32,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=64)smallDepthSort<64,1><<<1,64,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=128)smallDepthSort<128,1><<<1,128,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=256)smallDepthSort<128,2><<<1,128,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(build_mode_>=12)blockDepthSort512Pending<<<(capacity+511)/512,128,0,capture>>>(capacity,key_in,merge_in,key_out,sorted,capacity<=512,inverse,inputs);
    else blockDepthSort512<<<(capacity+511)/512,128,0,capture>>>(capacity,key_in,merge_in,key_out,sorted,capacity<=512,inverse);
    for(int width=512;width<capacity;width*=2) {
      if(build_mode_>=12)mergeDepthSortPending<<<(capacity+255)/256,256,0,capture>>>(capacity,width,merge_in,merge_out,key_out,sorted,width*2>=capacity,inverse,inputs);
      else mergeDepthSort<<<(capacity+255)/256,256,0,capture>>>(capacity,width,merge_in,merge_out,key_out,sorted,width*2>=capacity,inverse);
      std::swap(merge_in,merge_out);
    }
    if(build_mode_==7||build_mode_==8) {
      auto rectangles=reinterpret_cast<uint4*>(pool_metadata.data_ptr());
      auto limits=reinterpret_cast<float*>(rectangles+capacity);
      prepareFootprintMetadata<<<(capacity+127)/128,128,0,capture>>>(capacity,xy,conic,radii,sorted,key_out,touched,
          rectangles,limits,bits+int64_t(tiles)*words,grid);
      if(build_mode_==7)buildCompactedTiles<false><<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,bits,words,directory,groups,lists,counts,list_capacity,grid);
      else buildCompactedTiles<true><<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,bits,words,directory,groups,lists,counts,list_capacity,grid,inputs);
    } else {
      if(build_mode_!=6&&build_mode_!=9&&!fused_tasks) {
        if(device_clear_)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);
        else C10_CUDA_CHECK(cudaMemsetAsync(bits,0,bit_bytes,capture));
      }
      if(build_mode_==0) {
    if(ordered_coverage_) {
      scheduleBits<false,true><<<(capacity+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,tasks,task_count);
      executeBitTasks<false,true><<<workers,128,0,capture>>>(tasks,task_count,sorted,xy,conic,radii,bits,words,grid);
    } else {
      scheduleBits<false><<<(capacity+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,tasks,task_count);
      executeBitTasks<false><<<workers,128,0,capture>>>(tasks,task_count,sorted,xy,conic,radii,bits,words,grid);
    }
      }
      else if(build_mode_==1)buildGaussianFootprint<128,1><<<capacity,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==2)buildGaussianFootprint<64,1><<<capacity,64,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==3)buildGaussianFootprint<256,1><<<capacity,256,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==4)buildGaussianFootprint<128,2><<<capacity*2,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==5)buildGaussianFootprint<128,4><<<capacity*4,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==6)buildTileWords<false><<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==9)buildTileWords<true><<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid,inputs);
      else if(build_mode_==10||build_mode_>=12)executePreprocessedTasks<32><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid);
      else if(build_mode_==11)executePreprocessedTasks<128><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid);
      if(build_mode_==13||build_mode_==15||build_mode_>=16)compactActivePendingBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity,capacity,key_out,touched,inputs);
      else if(build_mode_>=12)compactPendingBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity,capacity,key_out,touched,inputs);
      else compactTileBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity,capacity,key_out,touched,pixel_mode_>=18?inputs:nullptr);
    }
    if(pixel_mode_>=6) {
      if(pixel_mode_==6)renderScalingPixels<1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==7)renderScalingPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==8)renderScalingPixels<3><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==9)renderScalingPixels<4><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==10)renderScalingPixels<5><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==11)renderScalingPixels<6><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==12)renderScalingPixels<7><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==13)renderScalingPixels<8><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==14)renderScalingPixels<9><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==18)renderScalingPixels<9><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==19)renderApproxPixels<1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==20)renderApproxPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==21)renderApproxPixels<3><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==22)renderApproxPixels<4><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==23)renderApproxPixels<5><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==24)renderApproxPixels<6><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==25)renderApproxPixels<7><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==26)renderApproxPixels<8><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==27)renderApproxPixels<9><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==28)renderApproxPixels<10><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==29)renderApproxPixels<11><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==30)renderApproxPixels<12><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==31)renderApproxPixels<13><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==32)renderApproxPixels<14><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==33)renderFoldedPixels<15><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==34)renderFoldedPixels<16><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==35)renderFoldedPixels<17><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==36)renderFoldedPixels<18><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==37)renderFoldedPixels<19><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==38)renderFoldedPixels<20><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==39)renderFoldedPixels<21><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==40)renderFoldedPixels<22><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==41)renderApproxPixels<23><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==42)renderApproxPixels<24><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==43)renderApproxPixels<25><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==44)renderApproxPixels<26><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==45)renderApproxPixels<27><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==46)renderFoldedPixels<28><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==47)renderFoldedPixels<29><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==48)renderFoldedPixels<30><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==49)renderFoldedPixels<31><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==50)renderFoldedPixels<32><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==51)renderTensorPixels<33><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==52)renderTensorPixels<34><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==53)renderTensorPixels<35><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==54)renderTensorPixels<36><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==55)renderFoldedPixels<37><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==56)renderFoldedPixels<38><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==57)renderFoldedPixels<39><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==58)renderFoldedPixels<40><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==200)renderEncodedSignPixels<38,3,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==15)renderPackedPixels<8><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==16)renderPackedPixels<16><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==17)renderPackedPixels<32><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    } else if(pixel_mode_==4)renderDirectPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else if(pixel_mode_==5)renderDirectPixels<1><<<grid,dim3(16,16),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else if(pixel_mode_==1)renderWarpPixels<8><<<grid,dim3(16,2),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(pixel_mode_==2)renderWarpPixels<4><<<grid,dim3(16,4),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(pixel_mode_==3)renderWarpPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,key_out,inputs);
    else if(variant==5)renderMultiPixel<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,
        list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    else renderCached<1><<<grid,dim3(16,16),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,
        list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    C10_CUDA_CHECK(cudaStreamEndCapture(capture,&graph));
    if(input_in_graph) {
      size_t roots=1;C10_CUDA_CHECK(cudaGraphGetRootNodes(graph,&plan->input_node,&roots));
      TORCH_CHECK(roots==1,"Input update must be the unique graph root");
    }
    C10_CUDA_CHECK(cudaGraphInstantiate(&plan->execution,graph,nullptr,nullptr,0));
    if(input_in_graph)plan->definition=graph;else C10_CUDA_CHECK(cudaGraphDestroy(graph));
    C10_CUDA_CHECK(cudaStreamDestroy(capture));
    return plan;
  }

  void precompile(int64_t points,int W,int H,int variant,int list_capacity,int plan_granularity,
      const Tensor& exemplar,bool input_in_graph) {
    TORCH_CHECK((variant==1||variant==5)&&list_capacity>0&&list_capacity<=256,"Graph precompile config");
    reserve(points,W,H,plan_granularity,exemplar);
    c10::cuda::CUDAGuard guard(exemplar.device());
    std::vector<int> capacities;
    if(plan_granularity==32){
      for(int k=32;k<=std::min(512,pool_capacity);k+=32)capacities.push_back(k);
      for(int k=1024;k<=pool_capacity;k+=512)capacities.push_back(k);
    } else if(plan_granularity==512)for(int k=512;k<=pool_capacity;k+=512)capacities.push_back(k);
    else for(int k=512;k<=pool_capacity;k*=2)capacities.push_back(k);
    TORCH_CHECK(capacities.size()<=size_t(max_plans),"All-capacity precompile exceeds configured plan count");
    // 模型加载阶段只编译有界调度计划；不运行Gaussian/相机计算，不读取任何视角内容。
    lookup.clear();plans.clear();
    for(int capacity:capacities) {
      auto plan=create(capacity,W,H,variant,list_capacity,exemplar.options(),input_in_graph);
      auto selected=plan.get();selected->last_used=++clock;
      plans.push_back(std::move(plan));lookup[std::make_tuple(capacity,W,H,variant,list_capacity,input_in_graph)]=selected;
    }
  }

  std::vector<Tensor> render(const Tensor& bg,const Tensor& xyz,const Tensor& colors,const Tensor& opacity,
      const Tensor& scales,const Tensor& rotation,double modifier,const Tensor& covariance,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
      const Tensor& dc,const Tensor& sh,int64_t degree,const Tensor& camera,int variant,int list_capacity,int plan_granularity,bool input_in_graph) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Capacity graph is frozen inference only");
    TORCH_CHECK(xyz.is_cuda()&&xyz.dim()==2&&xyz.size(1)==3&&H>0&&W>0,"Graph inputs");
    TORCH_CHECK((variant==1||variant==5)&&list_capacity>0&&list_capacity<=256,"Graph rendering config");
    TORCH_CHECK(plan_granularity==0||plan_granularity==512||plan_granularity==32,"Graph capacity granularity");
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
    auto background=bg.contiguous();
    auto x=xyz.contiguous(),s=scales.contiguous(),q=rotation.contiguous(),a=opacity.contiguous(),d=dc.contiguous(),h=sh.contiguous();
    auto color=colors.contiguous(),cov=covariance.contiguous();
    if(!parameters.defined())parameters=torch::empty({int64_t(sizeof(GraphInputs))},xyz.options().dtype(at::kByte));
    int capacity=plan_granularity==32?32:512;
    if(plan_granularity==32)capacity=P<=512?((P+31)/32)*32:((P+511)/512)*512;
    else if(plan_granularity==512)capacity=((P+511)/512)*512;
    else while(capacity<P)capacity*=2;
    reserve_pool(capacity,W,H,xyz.options());
    Key key=std::make_tuple(capacity,int(W),int(H),variant,list_capacity,input_in_graph);
    auto found=lookup.find(key);CapturedPlan* selected=found==lookup.end()?nullptr:found->second;
    if(!selected) {
      ++misses;
      while(plans.size()>=size_t(max_plans)) {
        auto oldest=std::min_element(plans.begin(),plans.end(),[](const auto& l,const auto& r){return l->last_used<r->last_used;});
        lookup.erase(std::make_tuple((*oldest)->capacity,(*oldest)->W,(*oldest)->H,(*oldest)->variant,(*oldest)->list_capacity,(*oldest)->input_in_graph));
        plans.erase(oldest);++evictions;
      }
      auto plan=create(capacity,W,H,variant,list_capacity,xyz.options(),input_in_graph);
      selected=plan.get();plans.push_back(std::move(plan));lookup[key]=selected;
    } else ++hits;
    selected->last_used=++clock;
    GraphInputs inputs{};inputs.P=P;inputs.D=degree;inputs.M=sh.numel()?sh.size(1):0;inputs.W=W;inputs.H=H;
    inputs.focal_x=W/(2.0f*float(tanx));inputs.focal_y=H/(2.0f*float(tany));inputs.tanx=tanx;inputs.tany=tany;inputs.modifier=modifier;
    inputs.xyz=x.data_ptr<float>();inputs.scales=s.data_ptr<float>();inputs.rotations=q.data_ptr<float>();inputs.opacity=a.data_ptr<float>();
    inputs.dc=d.data_ptr<float>();inputs.sh=h.data_ptr<float>();inputs.covariance=cov.numel()?cov.data_ptr<float>():nullptr;
    inputs.colors=color.numel()?color.data_ptr<float>():nullptr;
    inputs.view=view.data_ptr<float>();inputs.projection=proj.data_ptr<float>();inputs.camera=camera.data_ptr<float>();inputs.background=bg.data_ptr<float>();
    inputs.background=background.data_ptr<float>();
    inputs.v0=view.stride(0);inputs.v1=view.stride(1);inputs.p0=proj.stride(0);inputs.p1=proj.stride(1);inputs.c0=camera.stride(0);
    inputs.image=out.data_ptr<float>();inputs.radii=output_radii.data_ptr<int>();
    if(input_in_graph) {
      auto destination=reinterpret_cast<GraphInputs*>(parameters.data_ptr());void* arguments[]={&inputs,&destination};
      cudaKernelNodeParams config{};config.func=reinterpret_cast<void*>(updateGraphInputs);
      config.gridDim=dim3(1);config.blockDim=dim3(1);config.kernelParams=arguments;
      C10_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(selected->execution,selected->input_node,&config));
    } else updateGraphInputs<<<1,1,0,stream>>>(inputs,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));
    C10_CUDA_CHECK(cudaGraphLaunch(selected->execution,stream));C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,output_radii};
  }
  std::vector<Tensor> render_pending(const Tensor& bg,const Tensor& flat,const Tensor& frame,int64_t capacity_bound,int64_t rest,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,int64_t degree,const Tensor& camera){
    const Tensor& xyz=flat;const int variant=5,list_capacity=256,plan_granularity=32;const bool input_in_graph=false;const double modifier=1.;
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Capacity graph is frozen inference only");
    TORCH_CHECK(flat.is_cuda()&&flat.dim()==1&&flat.is_contiguous()&&H>0&&W>0&&capacity_bound>=0&&rest>=0&&flat.numel()==capacity_bound*(17+rest),"Pending flat attributes");
    TORCH_CHECK(frame.is_cuda()&&frame.device()==flat.device()&&frame.scalar_type()==at::kByte&&frame.numel()==sizeof(DeploymentFrame),"Pending GPU count layout");
    TORCH_CHECK((variant==1||variant==5)&&list_capacity>0&&list_capacity<=256,"Graph rendering config");
    TORCH_CHECK(plan_granularity==0||plan_granularity==512||plan_granularity==32,"Graph capacity granularity");
    c10::cuda::CUDAGuard guard(xyz.device());auto stream=c10::cuda::getCurrentCUDAStream();
    if(device>=0)TORCH_CHECK(device==xyz.get_device()&&owner==stream.stream(),"Use one graph renderer per device/stream");
    device=xyz.get_device();owner=stream.stream();
    for(const auto& t:{bg,flat,view,proj,camera})
      TORCH_CHECK(t.is_cuda()&&t.device()==xyz.device()&&t.scalar_type()==at::kFloat,"Input CUDA FP32/device");
    TORCH_CHECK(bg.numel()==3&&view.dim()==2&&view.size(0)==4&&view.size(1)==4&&proj.dim()==2&&proj.size(0)==4&&proj.size(1)==4&&camera.numel()==3,"Graph camera dimensions");
    const int P=capacity_bound;
    auto out=P?torch::empty({3,H,W},xyz.options()):torch::zeros({3,H,W},xyz.options());
    auto output_radii=P?torch::empty({P},xyz.options().dtype(at::kInt)):torch::zeros({P},xyz.options().dtype(at::kInt));
    if(!P)return {bg.view({3,1,1}).expand({3,H,W}).clone(),output_radii};
    auto background=bg.contiguous();
    if(!parameters.defined())parameters=torch::empty({int64_t(sizeof(GraphInputs))},xyz.options().dtype(at::kByte));
    int capacity=plan_granularity==32?32:512;
    if(plan_granularity==32)capacity=P<=512?((P+31)/32)*32:((P+511)/512)*512;
    else if(plan_granularity==512)capacity=((P+511)/512)*512;
    else while(capacity<P)capacity*=2;
    reserve_pool(capacity,W,H,xyz.options());
    Key key=std::make_tuple(capacity,int(W),int(H),variant,list_capacity,input_in_graph);
    auto found=lookup.find(key);CapturedPlan* selected=found==lookup.end()?nullptr:found->second;
    if(!selected) {
      ++misses;
      while(plans.size()>=size_t(max_plans)) {
        auto oldest=std::min_element(plans.begin(),plans.end(),[](const auto& l,const auto& r){return l->last_used<r->last_used;});
        lookup.erase(std::make_tuple((*oldest)->capacity,(*oldest)->W,(*oldest)->H,(*oldest)->variant,(*oldest)->list_capacity,(*oldest)->input_in_graph));
        plans.erase(oldest);++evictions;
      }
      auto plan=create(capacity,W,H,variant,list_capacity,xyz.options(),input_in_graph);
      selected=plan.get();plans.push_back(std::move(plan));lookup[key]=selected;
    } else ++hits;
    selected->last_used=++clock;
    GraphInputs inputs{};inputs.P=P;inputs.D=degree;inputs.M=rest/3;inputs.W=W;inputs.H=H;
    inputs.focal_x=W/(2.0f*float(tanx));inputs.focal_y=H/(2.0f*float(tany));inputs.tanx=tanx;inputs.tany=tany;inputs.modifier=modifier;
    inputs.covariance=nullptr;inputs.colors=nullptr;
    inputs.view=view.data_ptr<float>();inputs.projection=proj.data_ptr<float>();inputs.camera=camera.data_ptr<float>();inputs.background=bg.data_ptr<float>();
    inputs.background=background.data_ptr<float>();
    inputs.v0=view.stride(0);inputs.v1=view.stride(1);inputs.p0=proj.stride(0);inputs.p1=proj.stride(1);inputs.c0=camera.stride(0);
    inputs.image=out.data_ptr<float>();inputs.radii=output_radii.data_ptr<int>();
    updatePendingInputs<<<1,1,0,stream>>>(inputs,flat.data_ptr<float>(),reinterpret_cast<const DeploymentFrame*>(frame.data_ptr()),rest,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));
    C10_CUDA_CHECK(cudaGraphLaunch(selected->execution,stream));C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,output_radii};
  }
  std::vector<Tensor> diagnose()const {
    TORCH_CHECK(!plans.empty()&&pixel_mode_>=55&&pixel_mode_<=200,"Diagnostic modes37..40 require prior current-frame render");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==owner,"Use one graph renderer per device/stream");
    auto found=std::max_element(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a->last_used<b->last_used;});
    const auto& p=**found;const int n=p.capacity,W=p.W,H=p.H,tiles=((W+15)/16)*((H+15)/16),words=(n+31)/32,groups=(words+31)/32;
    char* geom=reinterpret_cast<char*>(p.geometry.data_ptr());
    using CudaRasterizer::obtain;float* depths;bool* clamped;int* radii;float2* xy;float* covariance;float4* conic;float* rgb;uint32_t* touched;
    obtain(geom,depths,n,128);obtain(geom,clamped,n*3,128);obtain(geom,radii,n,128);obtain(geom,xy,n,128);
    obtain(geom,covariance,n*6,128);obtain(geom,conic,n,128);obtain(geom,rgb,n*3,128);obtain(geom,touched,n,128);
    char* sorting=reinterpret_cast<char*>(p.sort.data_ptr());uint32_t *ki,*ko,*ii,*sorted;
    obtain(sorting,ki,n,128);obtain(sorting,ko,n,128);obtain(sorting,ii,n,128);obtain(sorting,sorted,n,128);
    auto image=torch::empty({3,H,W},parameters.options().dtype(at::kFloat));
    auto flags=torch::zeros({n},parameters.options().dtype(at::kInt));auto interactions=torch::zeros({1},parameters.options().dtype(at::kLong));
    auto bits=reinterpret_cast<uint32_t*>(p.bits.data_ptr());auto directory=reinterpret_cast<uint32_t*>(p.directory.data_ptr());
    auto lists=reinterpret_cast<uint32_t*>(p.lists.data_ptr());auto counts=reinterpret_cast<uint32_t*>(p.counts.data_ptr());
    auto inputs=reinterpret_cast<const GraphInputs*>(parameters.data_ptr());auto stream=c10::cuda::getCurrentCUDAStream();dim3 grid((W+15)/16,(H+15)/16);
    if(pixel_mode_==55)renderDiagnosedPixels<37><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
    if(pixel_mode_==56)renderDiagnosedPixels<38><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
    if(pixel_mode_==57)renderDiagnosedPixels<39><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
    if(pixel_mode_==58)renderDiagnosedPixels<40><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
    if(pixel_mode_==200)renderEncodedSignDiagnosedPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};
  }
  std::vector<Tensor> snapshot()const {
    TORCH_CHECK(!plans.empty(),"No rendered plan");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==owner,"Use one graph renderer per device/stream");
    auto found=std::max_element(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a->last_used<b->last_used;});
    const auto& p=**found;const int n=p.capacity,tiles=((p.W+15)/16)*((p.H+15)/16),words=(n+31)/32,groups=(words+31)/32;
    auto take=[&](void* data,std::vector<int64_t> shape){return torch::from_blob(data,shape,p.bits.options().dtype(at::kInt)).clone();};
    char* sorting=reinterpret_cast<char*>(p.sort.data_ptr());uint32_t *ki,*ko,*ii,*sorted;uint64_t *mi,*mo;
    CudaRasterizer::obtain(sorting,ki,n,128);CudaRasterizer::obtain(sorting,ko,n,128);
    CudaRasterizer::obtain(sorting,ii,n,128);CudaRasterizer::obtain(sorting,sorted,n,128);
    return {take(p.bits.data_ptr(),{tiles,words}),take(p.directory.data_ptr(),{tiles,groups}),
        take(p.lists.data_ptr(),{tiles,p.list_capacity}),take(p.counts.data_ptr(),{tiles}),take(sorted,{n}),take(ko,{n})};
  }
  std::vector<int64_t> stats()const{return {int64_t(hits),int64_t(misses),int64_t(evictions),int64_t(plans.size()),workspace_bytes()};}
};

std::vector<Tensor> render_pending_shared(void* handle,const Tensor& bg,
    const Tensor& flat,const Tensor& frame,int64_t capacity,int64_t rest,
    const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
    int64_t degree,const Tensor& camera){
  auto& raster=*static_cast<std::shared_ptr<StudentTileClearRasterizer>*>(handle);
  return raster->render_pending(bg,flat,frame,capacity,rest,view,proj,tanx,tany,H,W,degree,camera);
}

void register_pending_raster(pybind11::module_& m) {
  pybind11::class_<StudentTileClearRasterizer,std::shared_ptr<StudentTileClearRasterizer>>(m,"PendingRasterizer").def(pybind11::init<int,bool,bool,int,int,int>(),pybind11::arg("max_plans")=512,pybind11::arg("ordered_coverage")=true,pybind11::arg("device_clear")=true,pybind11::arg("build_mode")=0,pybind11::arg("pixel_mode")=0,pybind11::arg("preprocess_threads")=256)
      .def("render",&StudentTileClearRasterizer::render,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("render_pending",&StudentTileClearRasterizer::render_pending,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("_shared_handle",[](std::shared_ptr<StudentTileClearRasterizer> self){
        auto* handle=new std::shared_ptr<StudentTileClearRasterizer>(std::move(self));
        return pybind11::capsule(handle,"mycartgs.student-tile-clear.shared.v1",[](PyObject* object){
          delete static_cast<std::shared_ptr<StudentTileClearRasterizer>*>(PyCapsule_GetPointer(object,"mycartgs.student-tile-clear.shared.v1"));
        });
      })
      .def("reserve",&StudentTileClearRasterizer::reserve,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("precompile",&StudentTileClearRasterizer::precompile,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("diagnose",&StudentTileClearRasterizer::diagnose).def("snapshot",&StudentTileClearRasterizer::snapshot).def("workspace_bytes",&StudentTileClearRasterizer::workspace_bytes).def("stats",&StudentTileClearRasterizer::stats);
}
