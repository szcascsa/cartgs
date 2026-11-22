template<int Mode,int Kind,int Layout> __global__ void renderEncodedHalf2DiagnosedPixels(int W,int H,const uint32_t* bits,int words,
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
  bool half_used=false;
  __half2 packed_T=half2Value((inside[0]?0x3c00u:0xbc00u)|((inside[1]?0x3c00u:0xbc00u)<<16));
  __half2 packed_C[3]={__float2half2_rn(0.f),__float2half2_rn(0.f),__float2half2_rn(0.f)};
  const int count=counts[tile];
  if(count<=capacity) {
    __shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ __half2 half_rgb[256][3];__shared__ unsigned valid[Mode>=28&&Mode<=32?256:1];__shared__ unsigned unsafe_tile;
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
        for(int ch=0;ch<3;++ch)safe=safe&&isfinite(colors[id*3+ch])&&fabsf(colors[id*3+ch])<=65504.f;
        if(!safe)atomicOr(&unsafe_tile,1u);
      }
      if constexpr(true){
        f.a.x*=12102203.161561486f;f.a.y*=12102203.161561486f;f.a.z*=12102203.161561486f;f.a.w*=12102203.161561486f;
        f.b.x*=12102203.161561486f;f.b.y=fmaf(f.b.y,12102203.161561486f,1064970947.f);
      }
      abc[j]=f.a;ef[j]=f.b;for(int ch=0;ch<3;++ch)half_rgb[j][ch]=__float2half2_rn(colors[id*3+ch]);
    }
    __syncthreads();
    if(Mode>=37&&unsafe_tile){
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
        const int id=lists[tile*capacity+j];float2 pix[2]={{float(x),float(y)},{float(x),float(y+Step)}};
        debugOriginalPair(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F,id,flags,&visits);
      }
    }else{
    half_used=true;
    #pragma unroll 32
    for(int j=0;j<count;++j){
      if((j&7)==0&&__all_sync(0xffffffff,(half2Bits(packed_T)&0x80008000u)==0x80008000u))break;
      encodedHalfPair<true>(abc[j],ef[j],half_rgb[j],lx,ly,packed_T,packed_C,lists[tile*capacity+j],flags,&visits);
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
          }else diagnosedEncodedSignPair<Kind,Step>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,T,C,id,flags,&visits);
        }
      }
    }
  }
  if(visits)atomicAdd(interactions,visits);
  if(half_used){
    for(int ch=0;ch<3;++ch){C[0][ch]=__low2float(packed_C[ch]);C[1][ch]=__high2float(packed_C[ch]);}
    T[0]=__low2float(packed_T);T[1]=__high2float(packed_T);
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+Step*p)*W+x]=C[p][ch]+fabsf(T[p])*bg[ch];
}