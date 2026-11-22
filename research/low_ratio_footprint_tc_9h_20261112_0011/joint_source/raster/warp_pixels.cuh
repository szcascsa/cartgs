// 每warp协作加载32个有序成员，多个独立像素链复用相同寄存器属性。
// 每个像素仍按原成员次序、FP32 power/exp/alpha/T/C表达式依次合成。
template<int Pixels> __global__ void renderWarpPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,const uint32_t* counts,
    int capacity,const float2* xy,const float4* conic,const float* rgb,const uint32_t* keys,const GraphInputs* dynamic) {
  constexpr int Rows=16/Pixels;
  const int tile=blockIdx.y*gridDim.x+blockIdx.x,lane=(threadIdx.y*blockDim.x+threadIdx.x)&31;
  const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  const float* bg=dynamic->background;float* out=dynamic->image;
  float T[Pixels],C[Pixels][3];float2 pix[Pixels];bool inside[Pixels],done[Pixels];
  #pragma unroll
  for(int p=0;p<Pixels;++p){pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;}
  if(keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity) {
    for(int begin=0;begin<count;begin+=32) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      float2 point{};float4 co{};float3 color{};
      if(begin+lane<count) {
        const int id=lists[tile*capacity+begin+lane];point=xy[id];co=conic[id];color={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};
      }
      for(int member=0;member<min(32,count-begin);++member) {
        float2 q={__shfl_sync(0xffffffff,point.x,member),__shfl_sync(0xffffffff,point.y,member)};
        float4 c={__shfl_sync(0xffffffff,co.x,member),__shfl_sync(0xffffffff,co.y,member),__shfl_sync(0xffffffff,co.z,member),__shfl_sync(0xffffffff,co.w,member)};
        float3 col={__shfl_sync(0xffffffff,color.x,member),__shfl_sync(0xffffffff,color.y,member),__shfl_sync(0xffffffff,color.z,member)};
        #pragma unroll
        for(int p=0;p<Pixels;++p)compositeValues(q,c,col,pix[p],done[p],T[p],C[p]);
      }
    }
  } else {
    for(int group=0;group<groups;++group) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        const int word=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+word];
        while(members) {
          const int rank=word*32+__ffs(members)-1;members&=members-1;const int id=sorted[rank];
          const float2 q=xy[id];const float4 c=conic[id];const float3 col={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};
          #pragma unroll
          for(int p=0;p<Pixels;++p)compositeValues(q,c,col,pix[p],done[p],T[p],C[p]);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}
