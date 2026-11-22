// 大footprint专用机制：Tile中心的有序尾部辐射近似，提前结束后续逐像素成员遍历。
// 输入身份、K、投影和排序不变；近似路径与逐位路径独立。
struct TailReverseCompose {
  __device__ float4 operator()(float4 a,float4 b)const{
    return {fmaf(b.w,a.x,b.x),fmaf(b.w,a.y,b.y),fmaf(b.w,a.z,b.z),b.w*a.w};
  }
};

template<int StopPercent,bool Diagnose=false>
__global__ void renderTailPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const uint32_t* depth_keys,const GraphInputs* inputs,
    unsigned* flags=nullptr,unsigned long long* interactions=nullptr,unsigned long long* tail_stats=nullptr,float* diagnostic_image=nullptr){
  const int tid=threadIdx.x,tile=blockIdx.y*gridDim.x+blockIdx.x;
  const int x=blockIdx.x*16+tid%16,y=blockIdx.y*16+tid/16;
  const float cx=blockIdx.x*16+7.5f,cy=blockIdx.y*16+7.5f,lx=tid%16-7.5f,ly=tid/16-7.5f;
  const float* bg=inputs->background;float* out=Diagnose?diagnostic_image:inputs->image;
  bool inside[2]={x<W&&y<H,x<W&&y+8<H},done[2]={!inside[0],!inside[1]};
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  unsigned long long visits=0,tail_pixels=0,tail_members=0;
  if(depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]){
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity){
    __shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256],tail[257];
    __shared__ unsigned unsafe_tile;
    using Scan=cub::BlockScan<float4,128>;__shared__ typename Scan::TempStorage temp;
    if(!tid)unsafe_tile=0;
    __syncthreads();
    for(int j=tid;j<count;j+=128){
      const int id=lists[tile*capacity+j];const auto f=foldGaussian(xy[id],conic[id],cx,cy);
      bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
      const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
      for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
      if(!safe)atomicOr(&unsafe_tile,1u);
      abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    if(unsafe_tile){
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
        const int id=lists[tile*capacity+j];float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
        if constexpr(Diagnose)debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
        else compositeScalingPair<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F);
      }
    }else{
      float4 suffix[2];
      #pragma unroll
      for(int q=0;q<2;++q){
        const int j=255-(tid*2+q);float alpha=0.f;
        if(j<count&&ef[j].y>=-5.541263545158426f){alpha=fminf(.99f,boundedExp<1>(ef[j].y));if(alpha<1.f/255.f)alpha=0.f;}
        suffix[q]=alpha>0.f?float4{rgb[j].x*alpha,rgb[j].y*alpha,rgb[j].z*alpha,1.f-alpha}:float4{0.f,0.f,0.f,1.f};
      }
      Scan(temp).InclusiveScan(suffix,suffix,TailReverseCompose());
      #pragma unroll
      for(int q=0;q<2;++q){const int j=255-(tid*2+q);
        if(j<count)tail[j]={fmaf(suffix[q].w,bg[0],suffix[q].x),fmaf(suffix[q].w,bg[1],suffix[q].y),fmaf(suffix[q].w,bg[2],suffix[q].z)};}
      if(!tid)tail[count]={bg[0],bg[1],bg[2]};
      __syncthreads();
      #pragma unroll 8
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
        if(done[0]&&done[1])continue;
        const int id=lists[tile*capacity+j];
        if constexpr(Diagnose)debugFoldedPair<37>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C,id,flags,&visits);
        else foldedPair<37>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C);
        #pragma unroll
        for(int p=0;p<2;++p)if(!done[p]&&T[p]<StopPercent*.01f){
          C[p][0]=fmaf(T[p],tail[j+1].x,C[p][0]);C[p][1]=fmaf(T[p],tail[j+1].y,C[p][1]);C[p][2]=fmaf(T[p],tail[j+1].z,C[p][2]);
          T[p]=0.f;done[p]=true;
          if constexpr(Diagnose){++tail_pixels;tail_members+=count-j-1;}
        }
      }
    }
  }else{
    for(int group=0;group<groups;++group){
      if(__all_sync(0xffffffff,done[0]&&done[1]))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty){
        const int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members){
          const int rank=w*32+__ffs(members)-1;members&=members-1;const int id=sorted[rank];
          if(done[0]&&done[1])continue;
          float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
          if constexpr(Diagnose)debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
          else compositeScalingPair<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F);
        }
      }
    }
  }
  if constexpr(Diagnose){if(visits)atomicAdd(interactions,visits);if(tail_pixels){atomicAdd(tail_stats,tail_pixels);atomicAdd(tail_stats+1,tail_members);}}
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*8)*W+x]=C[p][ch]+T[p]*bg[ch];
}
