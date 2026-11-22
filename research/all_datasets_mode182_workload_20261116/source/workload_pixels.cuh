
// Counter slots: scheduled pair slots, valid active tests, effective blends, fallback tests.
__device__ __forceinline__ void workloadBlend(float power,float3 color,float& T,float* C,
    bool inside,unsigned long long* work){
  ++work[0];if(inside&&T>=0.f)++work[1];
  const float alpha=encodedAlpha(power,T<0.f);
  const float weight=alpha*T;
  const float next=T-weight;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  if(inside&&weight>0.f)++work[2];
  T=__int_as_float(__float_as_int(next)|(next<.01f?int(0x80000000u):0));
}

template<int Kind,int Step> __device__ __forceinline__ void workloadPair(FoldedGaussian f,float3 color,
    float x,float y,float* T,float C[2][3],const bool* inside,unsigned long long* work){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);
  workloadBlend(power,color,T[0],C[0],inside[0],work);workloadBlend(next,color,T[1],C[1],inside[1],work);
}

__device__ __forceinline__ void workloadOriginal(float2 point,float4 co,float3 color,float2 pix,
    bool& done,float& T,float* C,bool inside,unsigned long long* work){
  ++work[0];if(done)return;
  if(inside){++work[1];++work[3];}
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
  if(power>0.0f)return;
  float alpha=min(0.99f,co.w*exp(power));if(alpha<1.0f/255.0f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  T=test_T;if(inside)++work[2];
}

__device__ __forceinline__ void workloadFallback(float2 xy,float4 conic,float3 rgb,float2* pix,
    float* T,float C[2][3],const bool* inside,unsigned long long* work){
  bool done[2]={T[0]<0.f,T[1]<0.f};T[0]=fabsf(T[0]);T[1]=fabsf(T[1]);
  #pragma unroll
  for(int p=0;p<2;++p)workloadOriginal(xy,conic,rgb,pix[p],done[p],T[p],C[p],inside[p],work);
  if(done[0])T[0]=-T[0];if(done[1])T[1]=-T[1];
}

template<int Mode,int Kind,int Layout> __global__ void renderWorkloadPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float*,float*,const uint32_t* depth_keys,const GraphInputs* inputs,float* image,int64_t* metrics) {
  const float* bg=inputs->background;float* out=image;unsigned long long work[4]={0,0,0,0};
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
    if(!tid)for(int c=0;c<12;++c)metrics[tile*12+c]=0;
    return;
  }
  const int count=counts[tile];
  if(count<=capacity) {
    __shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256];__shared__ unsigned valid[Mode>=28&&Mode<=32?256:1];__shared__ unsigned unsafe_tile;
    if(Mode>=37){if(!tid)unsafe_tile=0;__syncthreads();}
    for(int j=tid;j<count;j+=128){
      const int id=lists[tile*capacity+j];auto f=foldGaussian(xy[id],conic[id],cx,cy);
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
        workloadFallback(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,T,C,inside,work);
      }
    }else{
    #pragma unroll 32
    for(int j=0;j<count;++j){
      if((j&7)==0&&__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
      if constexpr(Kind<3){if(T[0]<0.f&&T[1]<0.f)continue;}
      if(Mode>=28&&Mode<=32&&!valid[j]){
        const float4 a=abc[j];const float2 b=ef[j];
        float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
        workloadFallback({a.x,a.y},{a.z,a.w,b.x,b.y},rgb[j],pix,T,C,inside,work);
      }else workloadPair<Kind,Step>({abc[j],ef[j]},rgb[j],lx,ly,T,C,inside,work);
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
          auto f=foldGaussian(xy[id],conic[id],cx,cy);
          bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
          const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
          for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
          if(Mode>=37||Mode>=28&&!safe){
            float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
            workloadFallback(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,T,C,inside,work);
          }else workloadPair<Kind,Step>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,T,C,inside,work);
        }
      }
    }
  }

  __shared__ unsigned long long reduction[6][128];
  for(int c=0;c<4;++c)reduction[c][tid]=work[c];
  reduction[4][tid]=(inside[0]&&fabsf(T[0])<1.f)+(inside[1]&&fabsf(T[1])<1.f);
  reduction[5][tid]=(inside[0]&&T[0]<0.f)+(inside[1]&&T[1]<0.f);
  __syncthreads();
  if(tid<6){unsigned long long sum=0;for(int i=0;i<128;++i)sum+=reduction[tid][i];
    constexpr int col[6]={2,3,4,5,8,9};metrics[tile*12+col[tid]]=sum;}
  if(!tid){
    const int valid_w=min(16,W-int(blockIdx.x)*16),valid_h=min(16,H-int(blockIdx.y)*16);
    metrics[tile*12+0]=count;metrics[tile*12+1]=int64_t(count)*valid_w*valid_h;
    metrics[tile*12+6]=count>capacity;metrics[tile*12+7]=count>0;
    metrics[tile*12+10]=int64_t(count)*count;metrics[tile*12+11]=count;
  }

  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+Step*p)*W+x]=C[p][ch]+fabsf(T[p])*bg[ch];
}
