template<int Batch,bool Transpose> __global__ void renderSharedAlphaPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,const uint32_t* counts,
    int capacity,const float2* xy,const float4* conic,const float* rgb,const uint32_t* keys,const GraphInputs* inputs) {
  const int tid=threadIdx.x,tile=blockIdx.y*gridDim.x+blockIdx.x;
  const int ox=blockIdx.x*16,oy=blockIdx.y*16,x=ox+tid%16,y=oy+tid/16;
  const bool inside=x<W&&y<H;bool done=!inside;float T=1.f,C[3]={0.f};
  const float2 pixel=make_float2(x,y);const float* bg=inputs->background;float* out=inputs->image;
  if(keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity) {
    __shared__ float2 cached_xy[256];__shared__ float4 cached_co[256];__shared__ float3 cached_rgb[256];
    __shared__ float alpha[Batch][256];
    for(int j=tid;j<count;j+=256){int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_co[j]=conic[id];
      cached_rgb[j]={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};}
    __syncthreads();
    for(int base=0;base<count;base+=Batch) {
      const int valid=min(Batch,count-base);
      #pragma unroll
      for(int j=0;j<Batch;++j) {
        const int member=Transpose?tid%Batch:j,p=Transpose?(tid/Batch+j*(256/Batch)):tid;
        const float2 point=make_float2(ox+p%16,oy+p/16);
        if(member<valid)alpha[member][p]=alphaValue(cached_xy[base+member],cached_co[base+member],point);
      }
      __syncthreads();
      #pragma unroll
      for(int j=0;j<Batch;++j)if(j<valid)compositeAlpha(alpha[j][tid],cached_rgb[base+j],done,T,C);
      if(__syncthreads_count(done)==256)break;
    }
  } else {
    for(int group=0;group<groups;++group) {
      if(__all_sync(0xffffffff,done))break;
      unsigned dir=directory[tile*groups+group];
      while(dir){int word=group*32+__ffs(dir)-1;dir&=dir-1;unsigned mask=bits[tile*words+word];
        while(mask){int rank=word*32+__ffs(mask)-1;mask&=mask-1;
          compositeOne(sorted[rank],pixel,xy,conic,rgb,done,T,C);}}
    }
  }
  if(inside)for(int ch=0;ch<3;++ch)out[ch*H*W+y*W+x]=C[ch]+T*bg[ch];
}
