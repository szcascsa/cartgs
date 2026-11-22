// 用T符号表示像素已结束；幅值仍用于最终背景合成，逐成员顺序不变。
template<int Kind,int Step,int Cut> __device__ __forceinline__ void diagnosedCutoffBlend(
    float power,float3 color,float& T,float* C,unsigned id,unsigned* flags,unsigned long long* visits){
  const float alpha=cutoffAlpha<Cut>(power,T<0.f);
  const float weight=alpha*T;
  const float next=Kind==1?T*(1.f-alpha):T-weight;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  if(weight>0.f){atomicOr(flags+id,1u);++*visits;}
  T=__int_as_float(__float_as_int(next)|(next<.01f?int(0x80000000u):0));
}

template<int Kind,int Step,int Cut> __device__ __forceinline__ void diagnosedCutoffPair(DebugFoldedGaussian f,float3 color,
    float x,float y,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);
  diagnosedCutoffBlend<Kind,Step,Cut>(power,color,T[0],C[0],id,flags,visits);diagnosedCutoffBlend<Kind,Step,Cut>(next,color,T[1],C[1],id,flags,visits);
}


template<int Mode,int Kind,int Layout,int Cut> __global__ void renderCutoffDiagnosedPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float*,float*,const uint32_t* depth_keys,const GraphInputs* inputs,float* debug_image,unsigned* flags,unsigned long long* interactions) {
  const float* bg=inputs->background;float* out=debug_image;unsigned long long visits=0;
  const int tile=blockIdx.y*gridDim.x+blockIdx.x,tid=threadIdx.y*blockDim.x+threadIdx.x;
  constexpr int Step=Layout==0?8:1;
  const int lane=tid&31,warp=tid>>5;
  const int px=Layout==0?int(threadIdx.x):Layout==1?(warp&1)*8+(lane&7):Layout==2?(lane&15):warp*4+(lane&3);
  const int py=Layout==0?int(threadIdx.y):Layout==1?(warp>>1)*8+(lane>>3)*2:Layout==2?warp*4+(lane>>4)*2:(lane>>2)*2;
  const int x=blockIdx.x*16+px,y=blockIdx.y*16+py;
  const float cx=blockIdx.x*16+7.5f,cy=blockIdx.y*16+7.5f;
  const float lx=px-7.5f,ly=py-7.5f;
  bool inside[2]={x<W&&y<H,x<W&&y+Step<H},done[2]={!inside[0],!inside[1]};
  float T[2]={inside[0]?1.f:-1.f,inside[1]?1.f:-1.f},C[2][3]={{0.f},{0.f}};
  if(depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+Step*p)*W+x]=bg[ch];
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
      if constexpr(true){
        f.a.x*=12102203.161561486f;f.a.y*=12102203.161561486f;f.a.z*=12102203.161561486f;f.a.w*=12102203.161561486f;
        f.b.x*=12102203.161561486f;f.b.y=fmaf(f.b.y,12102203.161561486f,1064970947.f);
      }
      abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    if(Mode>=37&&unsafe_tile){
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
        const int id=lists[tile*capacity+j];float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
        debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
      }
    }else{
    #pragma unroll 32
    for(int j=0;j<count;++j){
      if((j&7)==0&&__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
      if constexpr(Kind<3){if(T[0]<0.f&&T[1]<0.f)continue;}
      if(Mode>=28&&Mode<=32&&!valid[j]){
        const int id=lists[tile*capacity+j];const float4 a=abc[j];const float2 b=ef[j];
        float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
        debugOriginalPair({a.x,a.y},{a.z,a.w,b.x,b.y},rgb[j],pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
      }else diagnosedCutoffPair<Kind,Step,Cut>({abc[j],ef[j]},rgb[j],lx,ly,T,C,lists[tile*capacity+j],flags,&visits);
    }
  }
  } else {
    for(int group=0;group<groups;++group){
      if(__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty){
        const int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;
        unsigned members=bits[tile*words+w];
        while(members){
          const int rank=w*32+__ffs(members)-1;members&=members-1;const int id=sorted[rank];
          if constexpr(Kind<3){if(T[0]<0.f&&T[1]<0.f)continue;}
          auto f=debugFoldGaussian(xy[id],conic[id],cx,cy);
          bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
          const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
          for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
          if(Mode>=37||Mode>=28&&!safe){
            float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
            debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
          }else diagnosedCutoffPair<Kind,Step,Cut>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,T,C,id,flags,&visits);
        }
      }
    }
  }
  if(visits)atomicAdd(interactions,visits);
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+Step*p)*W+x]=C[p][ch]+fabsf(T[p])*bg[ch];
}