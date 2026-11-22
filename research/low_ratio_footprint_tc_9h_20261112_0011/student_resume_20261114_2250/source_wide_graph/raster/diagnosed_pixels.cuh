__device__ __forceinline__ void debugOriginalPair(float2 point,float4 co,float3 color,float2* pix,
    bool* done,float* T,float C[2][3],float bound,unsigned id,unsigned* flags,unsigned long long* visits){
  #pragma unroll
  for(int p=0;p<2;++p){const float before=T[p];compositeScaling<9>(point,co,color,pix[p],done[p],T[p],C[p],bound);
    if(T[p]!=before){atomicOr(flags+id,1u);++*visits;}}
}
// 推理近似候选：Tile局部多项式、双像素增量；FP32颜色/T，深度顺序不变。
template<int Method> __device__ __forceinline__ float debugBoundedExp(float power) {
  if(Method==1)return __int_as_float(__float2int_rz(fmaf(power,12102203.161561486f,1064970947.f)));
  if(Method==2) {
    float x=power*1.4426950408889634f,n=floorf(x),f=x-n;
    float poly=fmaf(f,fmaf(f,fmaf(f,.05550410866f,.2402265069f),.69314718056f),1.f);
    return poly*__int_as_float((int(n)+127)<<23);
  }
  return __expf(power);
}

struct DebugFoldedGaussian {float4 a;float2 b;};

__device__ __forceinline__ DebugFoldedGaussian debugFoldGaussian(float2 point,float4 co,float cx,float cy) {
  const float gx=point.x-cx,gy=point.y-cy;
  DebugFoldedGaussian f;
  f.a={-.5f*co.x,-co.y,-.5f*co.z,co.x*gx+co.y*gy};
  f.b={co.z*gy+co.y*gx,-.5f*(co.x*gx*gx+co.z*gy*gy)-co.y*gx*gy+logf(co.w)};
  return f;
}

template<int Mode> __device__ __forceinline__ void debugBlendFolded(
    float power,float3 color,bool& done,float& T,float* C,unsigned id,unsigned* flags,unsigned long long* visits) {
  if(done||power<-5.541263545158426f)return;
  constexpr int Method=Mode==19||Mode==20||Mode>=28&&Mode<=31||Mode>=37&&Mode<=39?1:Mode==21||Mode==22?2:0;
  const float alpha=min(.99f,debugBoundedExp<Method>(power));
  if(alpha<1.f/255.f)return;
  const float next=T*(1-alpha);if(next<.0001f){done=true;return;}
  if(Mode==30||Mode==31||Mode==32||Mode>=37){const float w=alpha*T;C[0]=fmaf(color.x,w,C[0]);C[1]=fmaf(color.y,w,C[1]);C[2]=fmaf(color.z,w,C[2]);}
  else {C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;}T=next;
  atomicOr(flags+id,1u);++*visits;
  constexpr float Stop=Mode==16||Mode==20||Mode==22||Mode==29||Mode==31||Mode==38?.01f:Mode==39?.02f:Mode==17?.02f:Mode==18?.003f:0.f;
  if(Stop>0.f&&T<Stop)done=true;
}

template<int Mode> __device__ __forceinline__ void debugFoldedPair(
    DebugFoldedGaussian f,float3 color,float x,float y,bool* done,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits) {
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+8.f*(f.a.y*x+f.a.z*(2.f*y+8.f)+f.b.x);
  debugBlendFolded<Mode>(power,color,done[0],T[0],C[0],id,flags,visits);
  debugBlendFolded<Mode>(next,color,done[1],T[1],C[1],id,flags,visits);
}

template<int Mode> __global__ void renderDiagnosedPixels(int W,int H,const uint32_t* bits,int words,
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
      }else debugFoldedPair<Mode>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C,lists[tile*capacity+j],flags,&visits);
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
          }else debugFoldedPair<Mode>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C,id,flags,&visits);
        }
      }
    }
  }
  if(visits)atomicAdd(interactions,visits);
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+8*p)*W+x]=C[p][ch]+T[p]*bg[ch];
}
