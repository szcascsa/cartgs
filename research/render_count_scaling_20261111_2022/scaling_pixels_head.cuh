#include <math_constants.h>
// Mode 1: 对有限物理opacity增加严格安全的exp早拒绝；保留原Alpha计算。
// Mode 2: 每Tile成员预计算保守log边界，在执行exp前拒绝远低于Alpha阈值的像素。
// Mode 3: 将两个独立像素的公共横向项交给编译器复用。
// Mode 4: 与Mode 1相同，额外按opacity分档收紧安全边界。
__device__ __forceinline__ float alphaPowerBound(float opacity) {
  return opacity>0.f&&opacity<=1.f?-logf(opacity*255.f)-.001f:-CUDART_INF_F;
}
template<int Mode> __device__ __forceinline__ void compositeScaling(
    float2 point,float4 co,float3 color,float2 pix,bool& done,float& T,float* C,float bound) {
  if(done)return;
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
  if(power>0.0f)return;
  if(Mode==1) {
    // exp(-6) < 1/255，即使exp舍入，也远离原Alpha阈值。
    if(power < -6.f && co.w>=0.f && co.w<=1.f)return;
  }
  if(Mode==2&&power<bound)return;
  if(Mode==4) {
    const float threshold=co.w<=.0625f?-3.f:co.w<=.125f?-4.f:co.w<=.5f?-5.f:-6.f;
    if(power < threshold && co.w>=0.f && co.w<=1.f)return;
  }
  float alpha=min(0.99f,co.w*exp(power));if(alpha<1.0f/255.0f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  T=test_T;
}

template<int Mode> __device__ __forceinline__ void compositeScalingPair(
    float2 point,float4 co,float3 color,float2* pix,bool* done,float* T,float C[2][3],float bound) {
  if(Mode==3) {
    const float dx=point.x-pix[0].x;
    const float xx=(co.x*dx)*dx,cross=co.y*dx;
    #pragma unroll
    for(int p=0;p<2;++p) {
      if(done[p])continue;
      const float dy=point.y-pix[p].y;
      const float power=-.5f*(xx+co.z*dy*dy)-cross*dy;
      if(power>0.f)continue;
      const float alpha=min(.99f,co.w*exp(power));
      if(alpha<1.f/255.f)continue;
      const float next=T[p]*(1-alpha);
      if(next<.0001f){done[p]=true;continue;}
      C[p][0]+=color.x*alpha*T[p];C[p][1]+=color.y*alpha*T[p];C[p][2]+=color.z*alpha*T[p];
      T[p]=next;
    }
  } else {
    #pragma unroll
    for(int p=0;p<2;++p)compositeScaling<Mode>(point,co,color,pix[p],done[p],T[p],C[p],bound);
  }
}
