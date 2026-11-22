"""从已验证常规核生成图入口；仅追加动态指针、容量padding和调度参数。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'source/sparse_inference.cu').read_text()
source=source[:source.index('using Tensor=torch::Tensor;')]
source=source.replace('#include "forward.h"','#include "forward.h"\n#include "graph_inputs.h"',1)
begin=source.index('template<int Cache> __global__ void renderCached(')
end=source.index('// 计时外诊断：',begin)
kernel=source[begin:end]
kernel=kernel.replace('const uint32_t* sorted_depth_keys) {','const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {',1)
kernel=kernel.replace('  int tile=','  if(dynamic){bg=dynamic->background;out=dynamic->image;}\n  int tile=',1)
source=source[:begin]+kernel+source[end:]
begin=source.index('template<int Pixels> __global__ void renderMultiPixel(')
kernel=source[begin:]
kernel=kernel.replace('const uint32_t* sorted_depth_keys) {','const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {',1)
kernel=kernel.replace('  constexpr int Rows=','  if(dynamic){bg=dynamic->background;out=dynamic->image;}\n  constexpr int Rows=',1)
source=source[:begin]+kernel
source+='\n__global__ void updateGraphInputs(GraphInputs value,GraphInputs* destination){if(!threadIdx.x)*destination=value;}\n'
(HERE/'source/graph_kernels.cuh').write_text(source)

forward=(HERE/'source/forward_sparse.cu').read_text()
forward=forward[:forward.index('// Main rasterization method.')]
forward=forward.replace('#include "auxiliary.h"','#include "auxiliary.h"\n#include "graph_inputs.h"',1)
forward=forward.replace('bool Metadata = false>','bool Metadata = false, bool Dynamic = false>',1)
old='int c0=0,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr) {'
new='int c0=0,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr,const GraphInputs* dynamic=nullptr) {'
assert forward.count(old)==1;forward=forward.replace(old,new,1)
old='  if(Strided) {'
new='''  const int padded_capacity=P;
  if(Dynamic) {
    P=dynamic->P;D=dynamic->D;M=dynamic->M;W=dynamic->W;H=dynamic->H;
    orig_points=dynamic->xyz;scales=reinterpret_cast<const glm::vec3*>(dynamic->scales);
    rotations=reinterpret_cast<const glm::vec4*>(dynamic->rotations);opacities=dynamic->opacity;
    dc=dynamic->dc;shs=dynamic->sh;cov3D_precomp=dynamic->covariance;colors_precomp=dynamic->colors;
    viewmatrix=dynamic->view;projmatrix=dynamic->projection;cam_pos=reinterpret_cast<const glm::vec3*>(dynamic->camera);
    focal_x=dynamic->focal_x;focal_y=dynamic->focal_y;tan_fovx=dynamic->tanx;tan_fovy=dynamic->tany;
    scale_modifier=dynamic->modifier;v0=dynamic->v0;v1=dynamic->v1;p0=dynamic->p0;p1=dynamic->p1;c0=dynamic->c0;
  }
  if(Strided) {'''
assert forward.count(old)==1;forward=forward.replace(old,new,1)
# 需要重赋值的少量标量仅去除形参const，不改任何数学表达式。
for phrase in ['const float scale_modifier','const int W','const float tan_fovx','const float focal_x']:
    forward=forward.replace(phrase,phrase.replace('const ',''))
old='  auto idx = cg::this_grid().thread_rank();\n  if (idx >= P) return;'
new='''  auto idx = cg::this_grid().thread_rank();
  if(Dynamic) {
    if(idx>=padded_capacity)return;
    radii[idx]=0;tiles_touched[idx]=0;depth_keys[idx]=0xffffffff;depth_ids[idx]=idx;
    if(idx<P)dynamic->radii[idx]=0;
  }
  if (idx >= P) return;'''
assert forward.count(old)==1;forward=forward.replace(old,new,1)
forward=forward.replace('  radii[idx] = my_radius;','  radii[idx] = my_radius;\n  if(Dynamic)dynamic->radii[idx]=my_radius;',1)
forward=forward.replace('  depths[idx] = p_view.z;',
    '  if(Dynamic&&colors_precomp!=nullptr)for(int ch=0;ch<C;++ch)rgb[idx*C+ch]=colors_precomp[idx*C+ch];\n  depths[idx] = p_view.z;',1)
forward+='''
void graph_preprocess(int capacity,const GraphInputs* inputs,int* radii,float2* xy,float* depths,
    float* covariance,float* rgb,float4* conic,bool* clamped,uint32_t* touched,
    uint32_t* depth_keys,uint32_t* depth_ids,dim3 grid,cudaStream_t stream) {
  preprocessCUDA<3,true,true,false,true><<<(capacity+255)/256,256,0,stream>>>(capacity,0,0,
      nullptr,nullptr,1.f,nullptr,nullptr,nullptr,nullptr,clamped,nullptr,nullptr,nullptr,nullptr,nullptr,
      0,0,0.f,0.f,0.f,0.f,radii,xy,depths,covariance,rgb,conic,grid,touched,false,
      depth_keys,depth_ids,0,0,0,0,0,nullptr,nullptr,inputs);
}
'''
(HERE/'source/graph_forward.cu').write_text(forward)
print('GRAPH_SOURCES_GENERATED')
