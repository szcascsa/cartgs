// 成员压缩与属性装载共用CTA；使用四Warp并行扫描，而非单Warp串行枚举。
template<int Unroll> __global__ void renderCompactPixel(int W,int H,uint32_t* bits,int words,
    uint32_t* directory,int groups,const uint32_t* sorted,uint32_t* global_lists,uint32_t* counts,
    int capacity,const float2* xy,const float4* conic,const float* rgb,const uint32_t* keys,
    const uint32_t* touched,const int* radii,int n,const GraphInputs* dynamic) {
  const int tid=threadIdx.x,tile=blockIdx.y*gridDim.x+blockIdx.x;
  const int x=blockIdx.x*16+tid%16,y=blockIdx.y*16+tid/16;
  const float* bg=dynamic->background;float* out=dynamic->image;
  float2 pix[2]={make_float2(x,y),make_float2(x,y+8)};
  bool inside[2]={x<W&&y<H,x<W&&y+8<H},done[2]={!inside[0],!inside[1]};
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  __shared__ uint32_t members[256],count;
  __shared__ float2 cached_xy[256];__shared__ float4 cached_co[256];__shared__ float3 cached_rgb[256];
  __shared__ bool background_copy;
  if(!tid)background_copy=directBackgroundCopy(n,gridDim.x,gridDim.y,sorted,keys,touched,radii,xy,conic,bg);
  __syncthreads();
  if(background_copy){
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=bg[ch];
    return;
  }
  using Scan=cub::BlockScan<unsigned,128>;
  __shared__ typename Scan::TempStorage storage;
  unsigned carry=0;
  for(int base=0;base<words;base+=128) {
    const int word=base+tid;
    unsigned mask=word<words?bits[tile*words+word]:0;
    unsigned number=__popc(mask),prefix=0,total=0;
    Scan(storage).ExclusiveSum(number,prefix,total);
    unsigned position=carry+prefix;
    for(unsigned m=mask;m;m&=m-1) {
      const int rank=word*32+__ffs(m)-1;
      if(position<256)members[position]=sorted[rank];
      ++position;
    }
    carry+=total;
    __syncthreads();
  }
  if(!tid){count=carry;counts[tile]=carry;}
  __syncthreads();
  if(count<=256) {
    for(int j=tid;j<int(count);j+=128) {
      const int id=members[j];cached_xy[j]=xy[id];cached_co[j]=conic[id];
      cached_rgb[j]={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};
    }
    __syncthreads();
    #pragma unroll Unroll
    for(int j=0;j<int(count);++j) {
      if((j&31)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
      const float2 point=cached_xy[j];const float4 co=cached_co[j];const float3 color=cached_rgb[j];
      #pragma unroll
      for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
    }
  } else {
    for(int word=0;word<words;++word) {
      if(__all_sync(0xffffffff,done[0]&&done[1]))break;
      unsigned mask=bits[tile*words+word];
      while(mask) {
        int rank=word*32+__ffs(mask)-1;mask&=mask-1;int id=sorted[rank];
        const float2 point=xy[id];const float4 co=conic[id];const float3 color={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};
        #pragma unroll
        for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
      }
    }
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=C[p][ch]+T[p]*bg[ch];
}
