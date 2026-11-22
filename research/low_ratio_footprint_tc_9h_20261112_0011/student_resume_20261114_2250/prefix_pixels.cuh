// 4个lane共同处理一个像素的8个连续深度成员；结合律合成，不重排Gaussian身份。
// FP32归约分组改变舍入，属于独立近似候选。
__device__ __forceinline__ float prefixAlpha(float power,bool done){
  if(done||power<-5.541263545158426f)return 0.f;
  const float alpha=fminf(.99f,boundedExp<1>(power));
  return alpha<1.f/255.f?0.f:alpha;
}

__device__ __forceinline__ float4 prefixCompose(float4 a,float4 b){
  return {fmaf(a.w,b.x,a.x),fmaf(a.w,b.y,a.y),fmaf(a.w,b.z,a.z),a.w*b.w};
}

template<int Parts,bool Diagnose=false>
__global__ void renderPrefixPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const uint32_t* depth_keys,const GraphInputs* inputs,unsigned* flags=nullptr,
    unsigned long long* interactions=nullptr,float* diagnostic_image=nullptr){
  constexpr int Warps=16/Parts,Threads=Warps*32;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,group=lane/4,t=lane%4;
  const int tile=blockIdx.y*gridDim.x+blockIdx.x;
  const float cx=blockIdx.x*16+7.5f,cy=blockIdx.y*16+7.5f;
  const float* bg=inputs->background;float* out=Diagnose?diagnostic_image:inputs->image;
  float T[Parts][2],C[Parts][2][3];bool done[Parts][2],inside[Parts][2];
  unsigned long long visits=0;
  #pragma unroll
  for(int q=0;q<Parts;++q){
    #pragma unroll
    for(int p=0;p<2;++p){
      const int x=blockIdx.x*16+group+p*8,y=blockIdx.y*16+warp*Parts+q;
      inside[q][p]=x<W&&y<H;done[q][p]=!inside[q][p];T[q][p]=1.f;
      C[q][p][0]=C[q][p][1]=C[q][p][2]=0.f;
    }
  }
  if(depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]){
    if(!t)for(int q=0;q<Parts;++q)for(int p=0;p<2;++p)if(inside[q][p])for(int ch=0;ch<3;++ch)
      out[ch*H*W+(blockIdx.y*16+warp*Parts+q)*W+blockIdx.x*16+group+p*8]=bg[ch];
    return;
  }
  const int count=counts[tile];
  __shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256];__shared__ unsigned unsafe;
  if(!tid)unsafe=0;
  __syncthreads();
  if(count<=capacity){
    for(int j=tid;j<count;j+=Threads){
      const int id=lists[tile*capacity+j];const auto f=foldGaussian(xy[id],conic[id],cx,cy);
      bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
      const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
      for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
      if(!safe)atomicOr(&unsafe,1u);
      abc[j]=f.a;ef[j]=f.b;rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
  }
  __syncthreads();
  if(count<=capacity&&!unsafe){
    for(int base=0;base<count;base+=8){
      bool finished=true;
      #pragma unroll
      for(int q=0;q<Parts;++q)finished=finished&&done[q][0]&&done[q][1];
      if(__all_sync(0xffffffff,finished))break;
      const int j0=base+t*2,j1=j0+1;
      #pragma unroll
      for(int q=0;q<Parts;++q){
        const float lx=group-7.5f,ly=warp*Parts+q-7.5f;
        float powers[2][2]{};
        #pragma unroll
        for(int j=0;j<2;++j){
          const int index=j?j1:j0;
          if(index<count){const float4 a=abc[index];const float2 b=ef[index];
            powers[j][0]=fmaf(a.x,lx*lx,fmaf(a.z,ly*ly,fmaf(a.y,lx*ly,fmaf(a.w,lx,fmaf(b.x,ly,b.y)))));
            powers[j][1]=powers[j][0]+8.f*(a.x*(2.f*lx+8.f)+a.y*ly+a.w);
          }
        }
        #pragma unroll
        for(int p=0;p<2;++p){
          float aa=j0<count?prefixAlpha(powers[0][p],done[q][p]):0.f;
          float ab=j1<count?prefixAlpha(powers[1][p],done[q][p]):0.f;
          float prefix=(1.f-aa)*(1.f-ab);
          #pragma unroll
          for(int delta=1;delta<4;delta*=2){const float prior=__shfl_up_sync(0xffffffff,prefix,delta,4);if(t>=delta)prefix*=prior;}
          float before=__shfl_up_sync(0xffffffff,prefix,1,4);if(!t)before=1.f;
          const float incoming=T[q][p]*before,after_a=incoming*(1.f-aa),after_b=after_a*(1.f-ab);
          bool cross=false;
          if(done[q][p]||incoming<.01f){aa=ab=0.f;}
          else if(after_a<.0001f){aa=ab=0.f;cross=true;}
          else if(after_a<.01f){ab=0.f;}
          else if(after_b<.0001f){ab=0.f;cross=true;}
          const float wa=aa,wb=(1.f-aa)*ab;
          const float3 a=j0<count?rgb[j0]:float3{0.f,0.f,0.f},b=j1<count?rgb[j1]:float3{0.f,0.f,0.f};
          float4 value={fmaf(b.x,wb,a.x*wa),fmaf(b.y,wb,a.y*wa),fmaf(b.z,wb,a.z*wa),(1.f-aa)*(1.f-ab)};
          #pragma unroll
          for(int delta=1;delta<4;delta*=2){
            const float4 prior={__shfl_up_sync(0xffffffff,value.x,delta,4),__shfl_up_sync(0xffffffff,value.y,delta,4),
                __shfl_up_sync(0xffffffff,value.z,delta,4),__shfl_up_sync(0xffffffff,value.w,delta,4)};
            if(t>=delta)value=prefixCompose(prior,value);
          }
          value={__shfl_sync(0xffffffff,value.x,group*4+3),__shfl_sync(0xffffffff,value.y,group*4+3),
              __shfl_sync(0xffffffff,value.z,group*4+3),__shfl_sync(0xffffffff,value.w,group*4+3)};
          const unsigned crossed=__ballot_sync(0xffffffff,cross)&(15u<<(group*4));
          if(!t){C[q][p][0]=fmaf(T[q][p],value.x,C[q][p][0]);C[q][p][1]=fmaf(T[q][p],value.y,C[q][p][1]);C[q][p][2]=fmaf(T[q][p],value.z,C[q][p][2]);}
          T[q][p]*=value.w;done[q][p]=done[q][p]||T[q][p]<.01f||crossed;
          if constexpr(Diagnose){
            if(aa>0.f){atomicOr(flags+lists[tile*capacity+j0],1u);++visits;}
            if(ab>0.f){atomicOr(flags+lists[tile*capacity+j1],1u);++visits;}
          }
        }
      }
    }
  }else{
    for(int range=0;range<(count<=capacity?1:groups);++range){
      unsigned nonempty=count<=capacity?1u:directory[tile*groups+range];
      while(nonempty){const int word=count<=capacity?0:range*32+__ffs(nonempty)-1;nonempty&=nonempty-1;
        unsigned members=count<=capacity?1u:bits[tile*words+word];
        int list_index=0;
        while(members){const int rank=count<=capacity?0:word*32+__ffs(members)-1;
          const int id=count<=capacity?lists[tile*capacity+list_index]:sorted[rank];
          if(!t)for(int q=0;q<Parts;++q)for(int p=0;p<2;++p){const float previous=T[q][p];
            compositeScaling<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},
                {float(blockIdx.x*16+group+p*8),float(blockIdx.y*16+warp*Parts+q)},done[q][p],T[q][p],C[q][p],-CUDART_INF_F);
            if constexpr(Diagnose)if(T[q][p]!=previous){atomicOr(flags+id,1u);++visits;}}
          if(count<=capacity){if(++list_index>=count)members=0;}else members&=members-1;
        }
      }
    }
  }
  if constexpr(Diagnose)if(visits)atomicAdd(interactions,visits);
  if(!t)for(int q=0;q<Parts;++q)for(int p=0;p<2;++p)if(inside[q][p])for(int ch=0;ch<3;++ch)
    out[ch*H*W+(blockIdx.y*16+warp*Parts+q)*W+blockIdx.x*16+group+p*8]=C[q][p][ch]+T[q][p]*bg[ch];
}
