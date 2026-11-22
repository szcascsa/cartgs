// 与F2特殊背景copy条件一致；只在负零/非normal背景执行有界全局证明。
__device__ bool directBackgroundCopy(int n,int gx,int gy,const uint32_t* sorted,
    const uint32_t* keys,const uint32_t* touched,const int* radii,const float2* xy,const float4* conic,
    const float* background) {
  if(keys[0]==0xffffffff)return true;
  bool ordinary=true;
  #pragma unroll
  for(int ch=0;ch<3;++ch) {
    unsigned b=__float_as_uint(background[ch]),m=b&0x7fffffffu;
    ordinary=ordinary&&(b==0||(m>=0x00800000u&&m<0x7f800000u));
  }
  if(ordinary)return false;
  unsigned total=0;
  for(int rank=0;rank<min(n,257)&&keys[rank]!=0xffffffff;++rank){
    total+=touched[sorted[rank]];if(total>256)return false;
  }
  if(total==0)return true;
  for(int rank=0;rank<min(n,257)&&keys[rank]!=0xffffffff;++rank) {
    const unsigned id=sorted[rank];uint2 lo,hi;getRect(xy[id],radii[id],lo,hi,dim3(gx,gy));
    float limit=logf(conic[id].w/(1.f/255.f));
    for(int y=lo.y;y<int(hi.y);++y)for(int x=lo.x;x<int(hi.x);++x)
      if(orderedTilePower(xy[id],conic[id],x,y)<=limit)return false;
  }
  return true;
}

template<int Batch> __global__ void renderDirectTiles(int n,int W,int H,
    const uint32_t* sorted,const uint32_t* keys,const uint32_t* touched,const int* radii,
    const float2* xy,const float4* conic,const float* rgb,const GraphInputs* inputs) {
  const int tid=threadIdx.y*16+threadIdx.x,tx=blockIdx.x,ty=blockIdx.y;
  const int x=tx*16+threadIdx.x,y=ty*16+threadIdx.y,gx=gridDim.x,gy=gridDim.y;
  const float* bg=inputs->background;float* out=inputs->image;
  float2 pix[2]={make_float2(x,y),make_float2(x,y+8)};
  bool inside[2]={x<W&&y<H,x<W&&y+8<H},done[2]={!inside[0],!inside[1]};
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  __shared__ bool copy_background;
  if(!tid)copy_background=directBackgroundCopy(n,gx,gy,sorted,keys,touched,radii,xy,conic,bg);
  __syncthreads();
  if(copy_background) {
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=bg[ch];
    return;
  }
  using Scan=cub::BlockScan<int,128>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ float2 cached_xy[Batch];__shared__ float4 cached_co[Batch];__shared__ float3 cached_rgb[Batch];
  for(int base=0;base<n;base+=Batch) {
    if(keys[base]==0xffffffff)break;
    const int rank=base+tid;int id=0,present=0;
    if(tid<Batch&&rank<n&&keys[rank]!=0xffffffff) {
      id=sorted[rank];uint2 lo,hi;getRect(xy[id],radii[id],lo,hi,dim3(gx,gy));
      if(tx>=lo.x&&tx<hi.x&&ty>=lo.y&&ty<hi.y) {
        const float limit=logf(conic[id].w/(1.f/255.f));
        present=orderedTilePower(xy[id],conic[id],tx,ty)<=limit;
      }
    }
    int position=0,count=0;Scan(scan).ExclusiveSum(present,position,count);
    if(present) {
      cached_xy[position]=xy[id];cached_co[position]=conic[id];
      cached_rgb[position]={rgb[id*3],rgb[id*3+1],rgb[id*3+2]};
    }
    __syncthreads();
    #pragma unroll 32
    for(int j=0;j<count;++j) {
      if(done[0]&&done[1])break;
      float2 point=cached_xy[j];float4 co=cached_co[j];float3 color=cached_rgb[j];
      #pragma unroll
      for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
    }
    if(__syncthreads_count(done[0]&&done[1])==128)break;
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=C[p][ch]+T[p]*bg[ch];
}
