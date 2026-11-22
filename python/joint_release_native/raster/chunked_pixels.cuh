template<int Pixels,int CacheSize> __global__ void renderChunkedPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys) {
  constexpr int Rows=16/Pixels,Threads=256/Pixels;
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
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
          #pragma unroll
          for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}

