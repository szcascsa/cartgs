template<int Kind> __device__ __forceinline__ void diagnosedBranchlessBlend(
    float power,float3 color,bool& done,float& T,float* C,unsigned id,unsigned* flags,unsigned long long* visits){
  float alpha;
  if constexpr(Kind==4){const float value=mergedFoldedAlpha(power,false);alpha=done?0.f:value;}
  else alpha=mergedFoldedAlpha(power,done);
  if constexpr(Kind==1){
    if(alpha==0.f)return;
    const float next=T*(1.f-alpha);if(next<.0001f){done=true;return;}
    const float weight=alpha*T;C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1],id,flags,visits);C[2]=fmaf(color.z,weight,C[2]);
    if(weight>0.f){atomicOr(flags+id,1u);++*visits;}
    T=next;if(T<.01f)done=true;
  }else{
    float next=T*(1.f-alpha),weight=alpha*T;
    if constexpr(Kind==2){const bool crossing=next<.0001f;done=done||crossing;weight=crossing?0.f:weight;next=crossing?T:next;}
    C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1],id,flags,visits);C[2]=fmaf(color.z,weight,C[2]);
    if(weight>0.f){atomicOr(flags+id,1u);++*visits;}
    T=next;if(T<.01f)done=true;
  }
}

template<int Kind> __device__ __forceinline__ void diagnosedBranchlessPair(DebugFoldedGaussian f,float3 color,
    float x,float y,bool* done,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+8.f*(f.a.y*x+f.a.z*(2.f*y+8.f)+f.b.x);
  diagnosedBranchlessBlend<Kind>(power,color,done[0],T[0],C[0],id,flags,visits);diagnosedBranchlessBlend<Kind>(next,color,done[1],T[1],C[1],id,flags,visits);
}

template<int Mode,int Kind> __global__ void renderBranchlessDiagnosedPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float*,float*,const uint32_t* depth_keys,const GraphInputs* inputs,float* debug_image,unsigned* flags,unsigned long long* interactions) {
  const float* bg=inputs->background;float* out=debug_image;unsigned long long visits=0;
  const int tile=blockIdx.y*gridDim.x+blockIdx.x,tid=threadIdx.y*blockDim.x+threadIdx.x;
  const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  const float cx=blockIdx.x*16+7.5f,cy=blockIdx.y*16+7.5f;
  const float lx=threadIdx.x-7.5f,ly=threadIdx.y-7.5f;
  bool inside[2]={x<W&&y<H,x<W&&y+8<H},done[2]={!inside[0],!inside[1]};
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  if(depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+8*p)*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity) {
    __shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256];__shared__ unsigned valid[Mode>=28&&Mode<=32?256:1];__shared__ unsigned unsafe_tile;
    if(Mode>=37){if(!tid)unsafe_tile=0;__syncthreads();}
    for(int j=tid;j<count;j+=128){
      const int id=lists[tile*capacity+j];auto f=debugFoldGaussian(xy[id],conic[id],cx,cy);
      if(Mode>=28&&Mode<=32){
        const float4 co=conic[id];bool safe=co.w>0.f&&co.w<=1.f;
        const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
        for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
        valid[j]=safe;
        if(!safe){f.a={xy[id].x,xy[id].y,co.x,co.y};f.b={co.z,co.w};}
      }
      if(Mode>=37){
        bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
        const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
        for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
        if(!safe)atomicOr(&unsafe_tile,1u);
      }
      abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    if(Mode>=37&&unsafe_tile){
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
        const int id=lists[tile*capacity+j];float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
        debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
      }
    }else{
    #pragma unroll 32
    for(int j=0;j<count;++j){
      if((j&7)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
      if(done[0]&&done[1])continue;
      if(Mode>=28&&Mode<=32&&!valid[j]){
        const int id=lists[tile*capacity+j];const float4 a=abc[j];const float2 b=ef[j];
        float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
        debugOriginalPair({a.x,a.y},{a.z,a.w,b.x,b.y},rgb[j],pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
      }else diagnosedBranchlessPair<Kind>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C,lists[tile*capacity+j],flags,&visits);
    }
  }
  } else {
    for(int group=0;group<groups;++group){
      if(__all_sync(0xffffffff,done[0]&&done[1]))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty){
        const int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;
        unsigned members=bits[tile*words+w];
        while(members){
          const int rank=w*32+__ffs(members)-1;members&=members-1;const int id=sorted[rank];
          if(done[0]&&done[1])continue;
          auto f=debugFoldGaussian(xy[id],conic[id],cx,cy);
          bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
          const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
          for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
          if(Mode>=37||Mode>=28&&!safe){
            float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
            debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
          }else diagnosedBranchlessPair<Kind>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C,id,flags,&visits);
        }
      }
    }
  }
  if(visits)atomicAdd(interactions,visits);
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+8*p)*W+x]=C[p][ch]+T[p]*bg[ch];
}
