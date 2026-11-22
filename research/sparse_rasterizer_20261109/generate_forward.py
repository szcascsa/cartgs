"""从独立参考快照生成Warp短列表候选；保留预处理和普通混合公式。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
src=(HERE/'source/reference/cuda_rasterizer/forward.cu').read_text()
src=src.replace('#include "forward.h"','#include "forward.h"\n#include <c10/cuda/CUDAStream.h>')
src=src.replace('template <uint32_t CHANNELS, bool TRAINING = true>',
                'template <uint32_t CHANNELS, bool TRAINING = true, bool SHORT = false>')
needle='  int toDo = range.y - range.x;'
assert src.count(needle)==1
code=r'''
  // 短列表按Warp广播，像素仍按同一Gaussian顺序执行原始FP32公式。
  if (SHORT && !TRAINING && toDo <= 32) {
    const int lane=(threadIdx.y*blockDim.x+threadIdx.x)&31;
    int id=0;
    float2 xy{};float4 co{};
    if(lane<toDo) {
      id=point_list[range.x+lane];
      xy=points_xy_image[id];co=conic_opacity[id];
    }
    float T=1.0f;float C[CHANNELS]={0};
    for(int j=0;j<toDo;++j) {
      const int coll_id=__shfl_sync(0xffffffff,id,j);
      const float2 mean={__shfl_sync(0xffffffff,xy.x,j),__shfl_sync(0xffffffff,xy.y,j)};
      const float4 con_o={__shfl_sync(0xffffffff,co.x,j),__shfl_sync(0xffffffff,co.y,j),
          __shfl_sync(0xffffffff,co.z,j),__shfl_sync(0xffffffff,co.w,j)};
      if(!done) {
        float2 d={mean.x-pixf.x,mean.y-pixf.y};
        float power=-0.5f*(con_o.x*d.x*d.x+con_o.z*d.y*d.y)-con_o.y*d.x*d.y;
        if(!(power>0.0f)) {
          float alpha=min(0.99f,con_o.w*exp(power));
          if(!(alpha<1.0f/255.0f)) {
            float test_T=T*(1-alpha);
            if(test_T<0.0001f)done=true;
            else {
              for(int ch=0;ch<CHANNELS;++ch)C[ch]+=features[coll_id*CHANNELS+ch]*alpha*T;
              T=test_T;
            }
          }
        }
      }
    }
    if(inside)for(int ch=0;ch<CHANNELS;++ch)out_color[ch*H*W+pix_id]=C[ch]+T*bg_color[ch];
    return;
  }
'''
src=src.replace(needle,needle+'\n'+code)
# 所有候选核显式遵守调用方CUDA Stream；参考源只读。
src=src.replace('<<<grid, block>>>','<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>')
src=src.replace('<<<(P + 255) / 256, 256>>>','<<<(P + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>')
src+=r'''
void sparse_render_short(const dim3 grid,const uint2* ranges,const uint32_t* ids,int W,int H,
    const float2* xy,const float* colors,const float4* conic,const float* background,float* out) {
  renderCUDA<3,false,true><<<grid,dim3(16,16),0,c10::cuda::getCurrentCUDAStream()>>>(
      ranges,ids,nullptr,nullptr,nullptr,nullptr,W,H,xy,colors,conic,nullptr,nullptr,
      nullptr,background,out,0,nullptr,nullptr,nullptr,1,0,0);
}
'''
(HERE/'source/forward_sparse.cu').write_text(src)

reference=(HERE/'source/reference/cuda_rasterizer/rasterizer_impl.cu').read_text()
helpers=reference[reference.index('__device__ inline float evaluate_opacity_factor'):reference.index('// Wrapper method to call auxiliary coarse frustum containment test.')]
emit=reference[reference.index('__global__ void duplicateWithKeys'):reference.index('// Check keys to see if it is at the start/end of one tile')]
(HERE/'source/reference_emit.cuh').write_text(helpers+'\n'+emit.replace('duplicateWithKeys','emitSerial'))

