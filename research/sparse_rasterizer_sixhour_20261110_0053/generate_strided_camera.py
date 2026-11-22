"""在原预处理Kernel内按stride搬运35个相机标量，删除独立contiguous核。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/'source/forward_sparse.cu';source=path.read_text()
assert 'sixhour_preprocess_strided' not in source
source=source.replace('template <int C, bool Keys = false>','template <int C, bool Keys = false, bool Strided = false>',1)
old='bool prefiltered,uint32_t* depth_keys=nullptr,uint32_t* depth_ids=nullptr) {'
new='bool prefiltered,uint32_t* depth_keys=nullptr,uint32_t* depth_ids=nullptr,\n  int v0=0,int v1=0,int p0=0,int p1=0,int c0=0) {'
assert source.count(old)==1;source=source.replace(old,new,1)
old='  auto idx = cg::this_grid().thread_rank();\n  if (idx >= P) return;'
new='''  if(Strided) {
    __shared__ float camera_values[35];
    int t=threadIdx.x;
    if(t<16)camera_values[t]=viewmatrix[(t/4)*v0+(t%4)*v1];
    else if(t<32){int j=t-16;camera_values[t]=projmatrix[(j/4)*p0+(j%4)*p1];}
    else if(t<35)camera_values[t]=reinterpret_cast<const float*>(cam_pos)[(t-32)*c0];
    __syncthreads();
    viewmatrix=camera_values;projmatrix=camera_values+16;
    cam_pos=reinterpret_cast<const glm::vec3*>(camera_values+32);
  }
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;'''
assert source.count(old)==1;source=source.replace(old,new,1)
start=source.index('void sixhour_preprocess_keys(');stop=source.index('\n}',start)+2
wrapper=source[start:stop].replace('sixhour_preprocess_keys','sixhour_preprocess_strided',1)
wrapper=wrapper.replace('uint32_t* depth_ids) {','uint32_t* depth_ids,int v0,int v1,int p0,int p1,int c0,bool keys) {',1)
begin=wrapper.index('  preprocessCUDA<NUM_CHAFFELS,true>');launch=wrapper[begin:wrapper.rindex('}')].strip()
launch=launch.replace('preprocessCUDA<NUM_CHAFFELS,true>','preprocessCUDA<NUM_CHAFFELS,true,true>',1)
launch=launch.replace('prefiltered,depth_keys,depth_ids);','prefiltered,depth_keys,depth_ids,v0,v1,p0,p1,c0);')
wrapper=wrapper[:begin]+'  if(keys)'+launch+'\n  else '+launch.replace('preprocessCUDA<NUM_CHAFFELS,true,true>','preprocessCUDA<NUM_CHAFFELS,false,true>')+'\n}'
source+='\n\n'+wrapper+'\n';path.write_text(source)
header=HERE/'source/fused_preprocess.h';header.write_text(header.read_text()+'\n'+wrapper[:wrapper.index('{')].strip()+';\n')
print('STRIDED_CAMERA_GENERATED')
