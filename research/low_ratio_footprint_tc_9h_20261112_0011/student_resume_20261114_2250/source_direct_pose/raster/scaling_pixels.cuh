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

template<int Mode> __global__ void renderScalingPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {
  if(dynamic){bg=dynamic->background;out=dynamic->image;}
  constexpr int Pixels=2,Rows=8,Threads=128;
  int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  float T[Pixels],C[Pixels][3];float2 pix[Pixels];bool done[Pixels],inside[Pixels];
  #pragma unroll
  for(int p=0;p<Pixels;++p) {
    pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];
    T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;
  }
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=bg[ch];
    return;
  }
  int count=counts[tile];
  if(count<=capacity) {
    __shared__ float cached_bound[Mode==2?256:1];__shared__ float2 cached_xy[256];__shared__ float4 cached_conic[256];__shared__ float3 cached_color[256];
    for(int j=thread;j<count;j+=Threads) {
      int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
      cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);
    }
    __syncthreads();
    constexpr int Unroll=Mode==5?2:Mode==6?4:Mode==7?8:Mode==8?16:Mode==9?32:1;
    #pragma unroll Unroll
    for(int j=0;j<count;++j) {
      if((j&31)==0) {
        bool all_done=true;
        #pragma unroll
        for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
        if(__all_sync(0xffffffff,all_done))break;
      }
      float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];float bound=Mode==2?cached_bound[j]:-CUDART_INF_F;
      compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);
    }
  } else {
    for(int group=0;group<groups;++group) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members) {
          int rank=w*32+__ffs(members)-1;members&=members-1;int id=sorted[rank];
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};float bound=-CUDART_INF_F;
          compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}

