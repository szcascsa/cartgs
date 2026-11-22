"""小请求使用相应规模的稳定整数排序；大请求沿用原有排序。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_v2').exists()
shutil.copytree(source,HERE/'source_v2')
shutil.copy2(HERE/'build/mycartgs_render_count_scaling_v1.so',HERE/'reports/candidates_v2.so')
kernels=source/'graph_kernels.cuh'
text=kernels.read_text()
text += """
// stable key=(原始FP32深度位, 原始输入行ID)，不改变同深度次序。
__global__ void warpDepthSort32(int n,const uint32_t* input,uint32_t* output,
    uint32_t* ids,uint32_t* inverse) {
  const unsigned lane=threadIdx.x;
  uint64_t key=lane<unsigned(n)?(uint64_t(input[lane])<<32)|lane:uint64_t(-1);
  for(int size=2;size<=32;size*=2)for(int distance=size/2;distance;distance/=2) {
    const uint64_t other=(uint64_t(__shfl_xor_sync(0xffffffff,uint32_t(key>>32),distance))<<32)|
        __shfl_xor_sync(0xffffffff,uint32_t(key),distance);
    const bool low=(lane&distance)==0,ascending=(lane&size)==0;
    if((low==ascending)?other<key:other>key)key=other;
  }
  if(lane<unsigned(n)){output[lane]=key>>32;ids[lane]=uint32_t(key);if(inverse)inverse[uint32_t(key)]=lane;}
}

template<int Threads,int Items> __global__ void smallDepthSort(int n,const uint32_t* input,
    uint32_t* output,uint32_t* ids,uint32_t* inverse) {
  using Sort=cub::BlockRadixSort<uint32_t,Threads,Items,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[Items],values[Items];
  #pragma unroll
  for(int j=0;j<Items;++j){int i=threadIdx.x*Items+j;keys[j]=i<n?input[i]:0xffffffff;values[j]=i;}
  Sort(storage).Sort(keys,values);
  #pragma unroll
  for(int j=0;j<Items;++j){int i=threadIdx.x*Items+j;
    if(i<n){output[i]=keys[j];ids[i]=values[j];if(inverse)inverse[values[j]]=i;}}
}
"""
kernels.write_text(text)
main=source/'graph_main.cu'
text=main.read_text().replace('pixel_mode<=12','pixel_mode<=14')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==12)renderScalingPixels<7>' in x)
extra='\n'.join(line.replace('==12','=='+str(mode)).replace('<7>','<'+str(mode-5)+'>') for mode in (13,14))
text=text.replace(line,line+'\n'+extra)
line=next(x for x in text.splitlines() if 'blockDepthSort512<<<' in x)
text=text.replace(line,"""    if(capacity<=32)warpDepthSort32<<<1,32,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=64)smallDepthSort<64,1><<<1,64,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=128)smallDepthSort<128,1><<<1,128,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else if(capacity<=256)smallDepthSort<128,2><<<1,128,0,capture>>>(capacity,key_in,key_out,sorted,inverse);
    else """+line.strip())
main.write_text(text)
pixels=source/'scaling_pixels.cuh'
text=pixels.read_text().replace('Mode==7?8:1','Mode==7?8:Mode==8?16:Mode==9?32:1')
pixels.write_text(text)
print('CANDIDATES_V3_READY',flush=True)
