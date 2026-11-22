template<int Across,int Down,int Unroll> __global__ void renderMacrotilePixels(int W,int H,
    const uint32_t* bits,int words,const uint32_t* directory,int groups,const uint32_t* sorted,
    const uint32_t* lists,const uint32_t* counts,int capacity,
    const float2* xy,const float4* conic,const float* colors,const uint32_t* keys,const GraphInputs* inputs) {
  constexpr int Tiles=Across*Down,Threads=128*Tiles;
  const int thread=threadIdx.x,lane=thread&31,local_tile=thread/128,local_thread=thread%128;
  const int gx=(W+15)/16,gy=(H+15)/16;
  const int tx=blockIdx.x*Across+local_tile%Across,ty=blockIdx.y*Down+local_tile/Across;
  const bool valid_tile=tx<gx&&ty<gy;const int tile=ty*gx+tx;
  const int x=tx*16+local_thread%16,y=ty*16+local_thread/16;
  float2 pix[2]={make_float2(x,y),make_float2(x,y+8)};
  bool inside[2]={x<W&&y<H,x<W&&y+8<H},done[2]={!inside[0],!inside[1]};
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};const float* bg=inputs->background;float* out=inputs->image;
  if(keys[0]==0xffffffff||bits[int64_t(gx)*gy*words+1]) {
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=bg[ch];
    return;
  }
  __shared__ uint32_t union_ids[256],union_masks[256],union_count;
  __shared__ float2 cached_xy[256];__shared__ float4 cached_co[256];__shared__ float3 cached_rgb[256];
  // 第一个Warp枚举相邻Tile的位表并集，始终按原稳定深度rank顺序。
  if(thread<32) {
    unsigned carry=0;
    for(int group=0;group<groups;++group) {
      const int word=group*32+lane;unsigned masks[Tiles],combined=0;
      #pragma unroll
      for(int t=0;t<Tiles;++t) {
        const int a=blockIdx.x*Across+t%Across,b=blockIdx.y*Down+t/Across;
        masks[t]=word<words&&a<gx&&b<gy?bits[(b*gx+a)*words+word]:0;combined|=masks[t];
      }
      unsigned number=__popc(combined),scan=number;
      for(int offset=1;offset<32;offset*=2){unsigned v=__shfl_up_sync(0xffffffff,scan,offset);if(lane>=offset)scan+=v;}
      unsigned position=carry+scan-number;
      for(unsigned left=combined;left;left&=left-1) {
        const int bit=__ffs(left)-1;
        if(position<256) {
          union_ids[position]=sorted[word*32+bit];unsigned membership=0;
          #pragma unroll
          for(int t=0;t<Tiles;++t)if(masks[t]&(1u<<bit))membership|=1u<<t;
          union_masks[position]=membership;
        }
        ++position;
      }
      carry+=__shfl_sync(0xffffffff,scan,31);
    }
    if(!lane)union_count=carry;
  }
  __syncthreads();
  if(union_count<=256) {
    for(int j=thread;j<int(union_count);j+=Threads) {
      const int id=union_ids[j];cached_xy[j]=xy[id];cached_co[j]=conic[id];
      cached_rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    #pragma unroll Unroll
    for(int j=0;j<int(union_count);++j) {
      if((j&31)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
      if(union_masks[j]&(1u<<local_tile)) {
        float2 point=cached_xy[j];float4 co=cached_co[j];float3 color=cached_rgb[j];
        #pragma unroll
        for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
      }
    }
  } else if(valid_tile) {
    // 并集溢出时原Tile的全部有序成员仍必须参与，不截断。
    const unsigned count=counts[tile];
    if(count<=unsigned(capacity)) {
      #pragma unroll 8
      for(int j=0;j<int(count);++j) {
        if((j&31)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
        int id=lists[tile*capacity+j];float2 point=xy[id];float4 co=conic[id];
        float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
        #pragma unroll
        for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
      }
    } else {
      for(int group=0;group<groups;++group) {
        if(__all_sync(0xffffffff,done[0]&&done[1]))break;
        unsigned dir=directory[tile*groups+group];
        while(dir) {
          int word=group*32+__ffs(dir)-1;dir&=dir-1;unsigned mask=bits[tile*words+word];
          while(mask) {
            int rank=word*32+__ffs(mask)-1;mask&=mask-1;int id=sorted[rank];
            float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
            #pragma unroll
            for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
          }
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=C[p][ch]+T[p]*bg[ch];
}
