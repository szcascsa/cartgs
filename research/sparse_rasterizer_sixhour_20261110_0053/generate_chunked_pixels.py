"""独立shared容量候选；只按同一Tile成员顺序分块装入，所有点数走同一算法。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'source/sparse_inference.cu').read_text()
start=source.index('template<int Pixels> __global__ void renderMultiPixel(')
end=source.index('// Tile成员只在本Block枚举一次',start)
kernel=source[start:end].replace('template<int Pixels>','template<int Pixels,int CacheSize>',1).replace('void renderMultiPixel','void renderChunkedPixels',1)
begin=kernel.index('  if(count<=capacity) {');end=kernel.index('  } else {',begin)
kernel=kernel[:begin]+'''  if(count<=capacity) {
    __shared__ float2 cached_xy[CacheSize];__shared__ float4 cached_conic[CacheSize];__shared__ float3 cached_color[CacheSize];
    for(int base=0;base<count;base+=CacheSize) {
      int batch=min(CacheSize,count-base);
      for(int j=thread;j<batch;j+=Threads) {
        int id=lists[tile*capacity+base+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
        cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
      }
      __syncthreads();
      for(int j=0;j<batch;++j) {
        if((j&31)==0) {
          bool all_done=true;
          #pragma unroll
          for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
          if(__all_sync(0xffffffff,all_done))break;
        }
        float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];
        #pragma unroll
        for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
      }
      if(base+CacheSize<count)__syncthreads();
    }
'''+kernel[end:]
(HERE/'source/chunked_pixels.cuh').write_text(kernel)
print('CHUNKED_PIXEL_KERNEL_GENERATED')
