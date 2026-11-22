// Copyright follows the copied GRAPHDECO/CaRtGS reference; see reference sources.
// 独立推理候选：稳定Warp展开、短列表混合、共享深度序列Tile Bitset。
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cfloat>
#include "forward.h"
#include "auxiliary.h"
#include "rasterizer_impl.h"
namespace cg=cooperative_groups;

void sparse_render_short(dim3,const uint2*,const uint32_t*,int,int,const float2*,const float*,const float4*,const float*,float*);

// 参考控制核及Tile有效性公式由只读快照逐字生成。
#include "reference_emit.cuh"
__device__ bool touches(float2 xy,float4 co,int x,int y,float limit) {
  glm::vec2 p;
  const float v=max_contrib_power_rect_gaussian_float<15,15>(co,xy,
      glm::vec2(x*16,y*16),glm::vec2((x+1)*16-1,(y+1)*16-1),p);
  return v<=limit;
}
__device__ uint64_t sentinel() {return (uint64_t(0xffffffff)<<32)|__float_as_uint(FLT_MAX);}

__global__ void emitWarp(int n,const float2* xy,const float4* co,const float* z,
    const uint32_t* offsets,uint64_t* keys,uint32_t* ids,const int* radii,dim3 grid) {
  const int lane=threadIdx.x&31,i=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  if(i>=n||radii[i]<=0)return;
  uint2 lo,hi;getRect(xy[i],radii[i],lo,hi,grid);
  const float2 point=xy[i];const float4 conic=co[i];const float limit=logf(conic.w/(1.0f/255.0f));
  const int w=hi.x-lo.x,area=w*(hi.y-lo.y);uint32_t out=i?offsets[i-1]:0,end=offsets[i];
  for(int begin=0;begin<area;begin+=32) {
    int k=begin+lane,x=lo.x+k%w,y=lo.y+k/w;
    const bool valid=k<area && touches(point,conic,x,y,limit);
    unsigned mask=__ballot_sync(0xffffffff,valid);
    if(valid) {
      const unsigned rank=__popc(mask&((1u<<lane)-1));
      keys[out+rank]=(uint64_t(y*grid.x+x)<<32)|__float_as_uint(z[i]);ids[out+rank]=i;
    }
    out+=__popc(mask);
  }
  for(uint32_t j=out+lane;j<end;j+=32){keys[j]=sentinel();ids[j]=0xffffffff;}
}

__global__ void rangesKernel(int n,const uint64_t* keys,uint2* ranges,int tiles) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  unsigned tile=keys[i]>>32;
  if(!i){if(tile<tiles)ranges[tile].x=0;}
  else {unsigned prev=keys[i-1]>>32;if(prev!=tile){if(prev<tiles)ranges[prev].y=i;if(tile<tiles)ranges[tile].x=i;}}
  if(i==n-1&&tile<tiles)ranges[tile].y=n;
}
__global__ void depthKeys(int n,const float* z,const int* radii,uint32_t* keys,uint32_t* ids) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){keys[i]=radii[i]>0?__float_as_uint(z[i]):0xffffffff;ids[i]=i;}
}
__global__ void buildBits(int n,const uint32_t* sorted,const float2* xy,const float4* co,
    const int* radii,uint32_t* bits,int words,dim3 grid) {
  int lane=threadIdx.x&31,rank=(blockIdx.x*blockDim.x+threadIdx.x)>>5;if(rank>=n)return;
  int i=sorted[rank];if(radii[i]<=0)return;uint2 lo,hi;getRect(xy[i],radii[i],lo,hi,grid);
  const float2 point=xy[i];const float4 conic=co[i];const float limit=logf(conic.w/(1.0f/255.0f));
  int w=hi.x-lo.x,area=w*(hi.y-lo.y);
  for(int k=lane;k<area;k+=32){int x=lo.x+k%w,y=lo.y+k/w;
    if(touches(point,conic,x,y,limit))atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));}
}
__global__ void renderBits(int n,int W,int H,const uint32_t* bits,int words,const uint32_t* sorted,
    const float2* xy,const float4* co,const float* colors,const float* bg,float* out) {
  int tile=blockIdx.y*gridDim.x+blockIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  const bool inside=x<W&&y<H;bool done=!inside;float T=1.0f,C[3]={0};
  float2 pix={(float)x,(float)y};
  for(int w=0;w<words;++w) {
    if(__all_sync(0xffffffff,done))break;
    unsigned mask=bits[tile*words+w];
    while(mask) {
      int rank=w*32+__ffs(mask)-1;mask&=mask-1;
      const int i=sorted[rank];float2 point=xy[i];float4 con_o=co[i];
      if(!done) {
        float2 d={point.x-pix.x,point.y-pix.y};
        float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
        if(power>0.0f)continue;
        float alpha=min(0.99f,con_o.w*exp(power));if(alpha<1.0f/255.0f)continue;
        float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;continue;}
        for(int ch=0;ch<3;++ch)C[ch]+=colors[i*3+ch]*alpha*T;
        T=test_T;
      }
    }
  }
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}
__global__ void fillBackground(int count,int pixels,const float* bg,float* out) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=bg[i/pixels];
}

using Tensor=torch::Tensor;
struct SparseRasterizer {
  Tensor geom_,bin_,image_,bits_,sort_;int device_=-1;cudaStream_t stream_=nullptr;
  size_t max_bytes_=256ull<<20;
  static char* storage(Tensor& t,size_t bytes,torch::TensorOptions options) {
    // Scratch扩容丢弃旧内容，避免resize_复制未使用/未初始化的旧临时字节。
    if(!t.defined()||bytes>t.storage().nbytes())t=torch::empty({int64_t(bytes)},options.dtype(at::kByte));
    else t.resize_({int64_t(bytes)});
    return reinterpret_cast<char*>(t.data_ptr());
  }
  template<class T> Tensor snapshot(T* ptr,std::vector<int64_t> shape,at::ScalarType dtype,Tensor owner) {
    return torch::from_blob(ptr,shape,[owner](void*){},owner.options().dtype(dtype)).clone();
  }
  std::vector<Tensor> render(const Tensor& bg,const Tensor& xyz,const Tensor& colors,const Tensor& opacity,
      const Tensor& scales,const Tensor& rotation,double modifier,const Tensor& covariance,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,
      const Tensor& dc,const Tensor& sh,int64_t degree,const Tensor& camera,bool prefiltered,bool debug,
      int mode,bool short_lists,bool inspect) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"SparseRasterizer is frozen inference only");
    TORCH_CHECK(mode>=0&&mode<=2&&H>0&&W>0&&xyz.dim()==2&&xyz.size(1)==3,"Raster input/strategy");
    c10::cuda::CUDAGuard guard(xyz.device());auto stream=c10::cuda::getCurrentCUDAStream();
    if(device_>=0)TORCH_CHECK(device_==xyz.get_device()&&stream_==stream.stream(),"Use one renderer per device/stream");
    device_=xyz.get_device();stream_=stream.stream();
    TORCH_CHECK(xyz.scalar_type()==at::kFloat&&bg.numel()==3,"FP32/RGB required");
    for(const auto& t:{bg,xyz,colors,opacity,scales,rotation,covariance,view,proj,dc,sh,camera})
      TORCH_CHECK(t.is_cuda()&&t.device()==xyz.device()&&t.scalar_type()==at::kFloat,"Input CUDA FP32/device");
    int P=xyz.size(0),M=sh.numel()?sh.size(1):0;
    auto out=torch::zeros({3,H,W},xyz.options()),radii=torch::zeros({P},xyz.options().dtype(at::kInt));
    if(!P)return {out,radii};
    // 同原入口保留contiguous转换；所有计算状态逐帧重建。
    auto b=bg.contiguous(),x=xyz.contiguous(),c=colors.contiguous(),a=opacity.contiguous(),s=scales.contiguous();
    auto q=rotation.contiguous(),cov=covariance.contiguous(),v=view.contiguous(),p=proj.contiguous();
    auto d=dc.contiguous(),h=sh.contiguous(),cam=camera.contiguous();
    dim3 grid((W+15)/16,(H+15)/16);int tiles=grid.x*grid.y;
    size_t scan_bytes=0;cub::DeviceScan::InclusiveSum(nullptr,scan_bytes,(uint32_t*)nullptr,(uint32_t*)nullptr,P,stream);
    size_t gbytes=size_t(P)*(4+3+4+8+24+16+12+4+4)+scan_bytes+2048;
    char* ptr=storage(geom_,gbytes,xyz.options());
    if(inspect)C10_CUDA_CHECK(cudaMemsetAsync(ptr,0,gbytes,stream));
    using CudaRasterizer::obtain;
    float* depths;bool* clamped;int* internal;float2* xy;float* cov3;float4* conic;float* rgb;uint32_t* touched;uint32_t* offsets;char* scan;
    obtain(ptr,depths,P,128);obtain(ptr,clamped,3*P,128);obtain(ptr,internal,P,128);obtain(ptr,xy,P,128);
    obtain(ptr,cov3,6*P,128);obtain(ptr,conic,P,128);obtain(ptr,rgb,3*P,128);obtain(ptr,touched,P,128);
    obtain(ptr,offsets,P,128);obtain(ptr,scan,scan_bytes,128);
    FORWARD::preprocess(P,degree,M,x.data_ptr<float>(),reinterpret_cast<glm::vec3*>(s.data_ptr<float>()),modifier,
        reinterpret_cast<glm::vec4*>(q.data_ptr<float>()),a.data_ptr<float>(),d.data_ptr<float>(),h.data_ptr<float>(),clamped,
        cov.numel()?cov.data_ptr<float>():nullptr,c.numel()?c.data_ptr<float>():nullptr,v.data_ptr<float>(),p.data_ptr<float>(),
        reinterpret_cast<glm::vec3*>(cam.data_ptr<float>()),W,H,W/(2.0f*float(tanx)),H/(2.0f*float(tany)),
        tanx,tany,radii.data_ptr<int>(),xy,depths,cov3,rgb,conic,grid,touched,prefiltered);
    const float* feature=c.numel()?c.data_ptr<float>():rgb;
    uint64_t* sorted_keys=nullptr;uint32_t* sorted_ids=nullptr;uint2* ranges=nullptr;uint32_t* bits=nullptr;int L=0,words=(P+31)/32;
    if(mode<2) {
      C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(scan,scan_bytes,touched,offsets,P,stream));
      C10_CUDA_CHECK(cudaMemcpyAsync(&L,offsets+P-1,sizeof(int),cudaMemcpyDeviceToHost,stream));C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      if(L>0) {
        size_t sort_bytes=0;cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,(uint64_t*)nullptr,(uint64_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,L,0,32+int(ceil(log2(double(tiles+1)))),stream);
        size_t nbytes=size_t(L)*24+sort_bytes+1024;TORCH_CHECK(nbytes<=max_bytes_,"Binning workspace exceeds 256 MiB");
        ptr=storage(bin_,nbytes,xyz.options());uint64_t* unsorted;uint32_t* ids;char* temp;
        obtain(ptr,unsorted,L,128);obtain(ptr,sorted_keys,L,128);obtain(ptr,ids,L,128);obtain(ptr,sorted_ids,L,128);obtain(ptr,temp,sort_bytes,128);
        if(mode==0)emitSerial<<<(P+255)/256,256,0,stream>>>(P,xy,conic,depths,offsets,unsorted,ids,radii.data_ptr<int>(),grid,nullptr);
        else emitWarp<<<(P+3)/4,128,0,stream>>>(P,xy,conic,depths,offsets,unsorted,ids,radii.data_ptr<int>(),grid);
        C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp,sort_bytes,unsorted,sorted_keys,ids,sorted_ids,L,0,32+int(ceil(log2(double(tiles+1)))),stream));
        ranges=reinterpret_cast<uint2*>(storage(image_,size_t(tiles)*sizeof(uint2),xyz.options()));
        C10_CUDA_CHECK(cudaMemsetAsync(ranges,0,size_t(tiles)*sizeof(uint2),stream));
        rangesKernel<<<(L+255)/256,256,0,stream>>>(L,sorted_keys,ranges,tiles);
        if(short_lists)sparse_render_short(grid,ranges,sorted_ids,W,H,xy,feature,conic,b.data_ptr<float>(),out.data_ptr<float>());
        else FORWARD::render(grid,dim3(16,16),ranges,sorted_ids,nullptr,nullptr,nullptr,nullptr,W,H,xy,feature,conic,
            nullptr,nullptr,nullptr,b.data_ptr<float>(),out.data_ptr<float>(),0,nullptr,nullptr,true);
      } else fillBackground<<<(W*H*3+255)/256,256,0,stream>>>(W*H*3,W*H,b.data_ptr<float>(),out.data_ptr<float>());
    } else {
      size_t bytes=size_t(tiles)*words*4;TORCH_CHECK(bytes<=max_bytes_,"Tile Bitset exceeds 256 MiB");
      bits=reinterpret_cast<uint32_t*>(storage(bits_,bytes,xyz.options()));C10_CUDA_CHECK(cudaMemsetAsync(bits,0,bytes,stream));
      size_t sort_bytes=0;cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes,(uint32_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,(uint32_t*)nullptr,P,0,32,stream);
      ptr=storage(sort_,size_t(P)*16+sort_bytes+1024,xyz.options());uint32_t *key_in,*key_out,*id_in;char* temp;
      obtain(ptr,key_in,P,128);obtain(ptr,key_out,P,128);obtain(ptr,id_in,P,128);obtain(ptr,sorted_ids,P,128);obtain(ptr,temp,sort_bytes,128);
      depthKeys<<<(P+255)/256,256,0,stream>>>(P,depths,radii.data_ptr<int>(),key_in,id_in);
      C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp,sort_bytes,key_in,key_out,id_in,sorted_ids,P,0,32,stream));
      buildBits<<<(P+3)/4,128,0,stream>>>(P,sorted_ids,xy,conic,radii.data_ptr<int>(),bits,words,grid);
      renderBits<<<grid,dim3(16,16),0,stream>>>(P,W,H,bits,words,sorted_ids,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(inspect) {
      std::vector<Tensor> result={out,radii,snapshot(xy,{P,2},at::kFloat,geom_),snapshot(conic,{P,4},at::kFloat,geom_),
          snapshot(depths,{P},at::kFloat,geom_),snapshot(touched,{P},at::kInt,geom_)};
      auto empty=torch::empty({0},xyz.options().dtype(at::kInt));
      result.push_back(sorted_keys?snapshot(sorted_keys,{L},at::kLong,bin_):empty);
      result.push_back(sorted_ids?snapshot(sorted_ids,{mode==2?P:L},at::kInt,mode==2?sort_:bin_):empty);
      result.push_back(ranges?snapshot(ranges,{tiles,2},at::kInt,image_):empty);
      result.push_back(bits?snapshot(bits,{tiles,words},at::kInt,bits_):empty);
      return result;
    }
    return {out,radii};
  }
  int64_t workspace_bytes() const {int64_t n=0;for(const auto& t:{geom_,bin_,image_,bits_,sort_})if(t.defined())n+=t.storage().nbytes();return n;}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<SparseRasterizer>(m,"Rasterizer").def(pybind11::init<>())
      .def("render",&SparseRasterizer::render,pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("workspace_bytes",&SparseRasterizer::workspace_bytes);
}
