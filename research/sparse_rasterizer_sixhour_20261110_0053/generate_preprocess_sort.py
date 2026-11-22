"""仅移动原预处理表达式：每512点Block先做几何，再以原CUB稳定排序。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'source/forward_sparse.cu').read_text()
start=source.index('// Forward method for converting')
begin=source.index('template <int C, bool Keys')
end=source.index('// Main rasterization method.',begin)
one=source[begin:end]
one=one.replace('__global__ void preprocessCUDA(int P,','__device__ __forceinline__ void preprocessOne(int idx,int P,',1)
camera_begin=one.index('  if(Strided) {');camera_end=one.index('  auto idx = cg::this_grid().thread_rank();',camera_begin)
camera_code=one[camera_begin:camera_end]
one=one[:camera_begin]+one[camera_end:]
one=one.replace('  auto idx = cg::this_grid().thread_rank();\n','',1)
kernel_signature=source[begin:end]
kernel_signature=kernel_signature[kernel_signature.index('__global__'):kernel_signature.index(') {')+1]
kernel_signature=kernel_signature.replace('preprocessCUDA','preprocessSortCUDA',1)
kernel_signature=kernel_signature[:-1]+',uint64_t* temporary=nullptr,uint32_t* sorted_keys=nullptr,uint32_t* sorted_ids=nullptr,bool final=false)'
kernel=kernel_signature+' {\n'+camera_code.replace('  if(Strided) {','  {',1)+'''
  #pragma unroll 1
  for(int item=0;item<4;++item) {
    int idx=blockIdx.x*512+threadIdx.x+item*128;
    preprocessOne<3,true,false,false>(idx,P,D,M,orig_points,scales,scale_modifier,rotations,opacities,dc,shs,
        clamped,cov3D_precomp,colors_precomp,viewmatrix,projmatrix,cam_pos,W,H,tan_fovx,tan_fovy,focal_x,focal_y,
        radii,points_xy_image,depths,cov3Ds,rgb,conic_opacity,grid,tiles_touched,prefiltered,depth_keys,depth_ids);
  }
  __syncthreads();
  using Sort=cub::BlockRadixSort<uint32_t,128,4,uint32_t>;
  __shared__ typename Sort::TempStorage storage;
  uint32_t keys[4],values[4];
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    keys[j]=i<unsigned(P)?depth_keys[i]:0xffffffff;values[j]=i;
  }
  Sort(storage).Sort(keys,values);
  for(int j=0;j<4;++j) {
    unsigned i=blockIdx.x*512+threadIdx.x*4+j;
    if(i<unsigned(P)) {
      if(final){sorted_keys[i]=keys[j];sorted_ids[i]=values[j];}
      else temporary[i]=(uint64_t(keys[j])<<32)|values[j];
    }
  }
}
'''
wrapper_start=source.index('void sixhour_preprocess_strided(')
signature=source[wrapper_start:source.index('{',wrapper_start)].strip()
signature=signature.replace('sixhour_preprocess_strided','sixhour_preprocess_sort',1)
signature=signature.replace(',bool keys)',',uint64_t* temporary,uint32_t* sorted_keys,uint32_t* sorted_ids,bool final)')
call_start=source.index('      P, D, M, means3D',wrapper_start)
call_end=source.index(';',call_start)
arguments=source[call_start:call_end].strip()
assert arguments.endswith('v0,v1,p0,p1,c0)')
arguments=arguments[:-1]+',nullptr,nullptr,temporary,sorted_keys,sorted_ids,final)'
wrapper=signature+' {\n  preprocess_sort_study::preprocessSortCUDA<<<(P+511)/512,128,0,c10::cuda::getCurrentCUDAStream()>>>('+arguments+';\n}\n'
output=source[:start]+'\nnamespace preprocess_sort_study {\n'+source[start:begin]+one+'\n'+kernel+'\n}\n'+wrapper
(HERE/'source/preprocess_sort.cu').write_text(output)
(HERE/'source/preprocess_sort.h').write_text('#pragma once\n'+signature+';\n')
print('PREPROCESS_SORT_GENERATED')
