// 只改Gaussian->Tile整数成员构建；所有覆盖公式复用原SASS固定的FP32运算顺序。
template<int Threads,int Shards> __global__ void buildGaussianFootprint(int n,const uint32_t* sorted,
    const float2* xy,const float4* conic,const int* radii,uint32_t* bits,int words,dim3 grid) {
  const int rank=blockIdx.x/Shards,shard=blockIdx.x%Shards;
  if(rank>=n)return;
  const int id=sorted[rank];if(radii[id]<=0)return;
  uint2 lo,hi;getRect(xy[id],radii[id],lo,hi,grid);
  const int width=hi.x-lo.x,area=width*(hi.y-lo.y);
  const float2 point=xy[id];const float4 co=conic[id];const float limit=logf(co.w/(1.0f/255.0f));
  for(int k=shard*Threads+threadIdx.x;k<area;k+=Threads*Shards) {
    const int x=lo.x+k%width,y=lo.y+k/width;
    if(orderedTilePower(point,co,x,y)<=limit)
      atomicOr(bits+(y*grid.x+x)*words+rank/32,1u<<(rank&31));
  }
}

// 每warp聚合一个Tile/深度字，消除同一个32位成员字的多Gaussian原子竞争。
template<bool Dynamic=false> __global__ void buildTileWords(int n,const uint32_t* sorted,const float2* xy,const float4* conic,
    const int* radii,uint32_t* bits,int words,dim3 grid,const GraphInputs* dynamic=nullptr) {
  const int tile=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const int tx=tile%grid.x,ty=tile/grid.x;
  const int active_words=Dynamic?(min(n,dynamic->P)+31)/32:words;
  for(int word=warp;word<active_words;word+=4) {
    const int rank=word*32+lane;bool present=false;
    if(rank<n) {
      const int id=sorted[rank];
      if(radii[id]>0) {
        uint2 lo,hi;getRect(xy[id],radii[id],lo,hi,grid);
        if(tx>=lo.x&&tx<hi.x&&ty>=lo.y&&ty<hi.y) {
          const float4 co=conic[id];const float limit=logf(co.w/(1.0f/255.0f));
          present=orderedTilePower(xy[id],co,tx,ty)<=limit;
        }
      }
    }
    unsigned mask=__ballot_sync(0xffffffff,present);
    if(!lane)bits[int64_t(tile)*words+word]=mask;
  }
  if(Dynamic)for(int word=active_words+threadIdx.x;word<words;word+=blockDim.x)bits[int64_t(tile)*words+word]=0;
  if(tile==0&&!threadIdx.x){bits[int64_t(grid.x)*grid.y*words]=0;bits[int64_t(grid.x)*grid.y*words+1]=0;}
}

// 预计算只在当前帧Workspace保存，每帧重算bbox和阈值；不改变预处理浮点表达式。
__global__ void prepareFootprintMetadata(int n,const float2* xy,const float4* conic,const int* radii,
    const uint32_t* sorted,const uint32_t* keys,const uint32_t* touched,uint4* rectangles,float* limits,
    uint32_t* flags,dim3 grid) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id<n) {
    uint2 lo=make_uint2(0,0),hi=make_uint2(0,0);float limit=0.f;
    if(radii[id]>0){getRect(xy[id],radii[id],lo,hi,grid);limit=logf(conic[id].w/(1.0f/255.0f));}
    rectangles[id]=make_uint4(lo.x,lo.y,hi.x,hi.y);limits[id]=limit;
  }
  if(blockIdx.x==0&&threadIdx.x<32) {
    const int lane=threadIdx.x;unsigned total=0;
    for(int base=0;base<min(n,257);base+=32) {
      const int rank=base+lane;
      unsigned value=rank<min(n,257)&&keys[rank]!=0xffffffff?touched[sorted[rank]]:0;
      for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
      total+=__shfl_sync(0xffffffff,value,0);
      if(total>256)break;
    }
    if(!lane){flags[0]=total<=256;flags[1]=total<=256;}
  }
}

// 一个warp负责一个Tile；ballot得到有序成员，直接写位表、目录、短列表和真实count。
// 溢出仍写完整位表和目录，像素核沿原深度次序回退枚举，不截断。
template<bool Dynamic=false> __global__ void buildCompactedTiles(int n,const uint32_t* sorted,const float2* xy,const float4* conic,
    const uint4* rectangles,const float* limits,uint32_t* bits,int words,uint32_t* directory,int groups,
    uint32_t* lists,uint32_t* counts,int list_capacity,dim3 grid,const GraphInputs* dynamic=nullptr) {
  const int lane=threadIdx.x&31,tile=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
  const int tiles=grid.x*grid.y;if(tile>=tiles)return;
  const unsigned tx=tile%grid.x,ty=tile/grid.x;
  unsigned carry=0,dir=0;const bool sparse_candidate=bits[int64_t(tiles)*words]!=0;
  const int active_words=Dynamic?(min(n,dynamic->P)+31)/32:words;
  for(int word=0;word<active_words;++word) {
    const int rank=word*32+lane;unsigned id=0;bool present=false;
    if(rank<n) {
      id=sorted[rank];const uint4 rect=rectangles[id];
      if(tx>=rect.x&&tx<rect.z&&ty>=rect.y&&ty<rect.w)
        present=orderedTilePower(xy[id],conic[id],tx,ty)<=limits[id];
    }
    const unsigned mask=__ballot_sync(0xffffffff,present);
    if(!lane)bits[int64_t(tile)*words+word]=mask;
    if(mask)dir|=1u<<(word&31);
    if(present) {
      const unsigned position=carry+__popc(mask&((1u<<lane)-1));
      if(position<unsigned(list_capacity))lists[tile*list_capacity+position]=id;
    }
    carry+=__popc(mask);
    if(!lane&&sparse_candidate&&mask)atomicExch(bits+int64_t(tiles)*words+1,0);
    if((word&31)==31||word==active_words-1) {
      if(!lane)directory[tile*groups+word/32]=dir;
      dir=0;
    }
  }
  if(Dynamic){
    for(int word=active_words+lane;word<words;word+=32)bits[int64_t(tile)*words+word]=0;
    for(int group=(active_words+31)/32+lane;group<groups;group+=32)directory[tile*groups+group]=0;
  }
  if(!lane)counts[tile]=carry;
}
