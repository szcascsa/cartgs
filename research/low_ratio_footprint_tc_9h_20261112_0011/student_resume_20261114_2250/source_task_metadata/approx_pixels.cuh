// 推理研究：FastExp与受控贡献/透射率阈值，Mode0仍由冻结F2作对照。
template<int Mode> __device__ __forceinline__ void compositeApprox(
    float2 point,float4 co,float3 color,float2 pix,bool& done,float& T,float* C,float) {
  if(done)return;
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
  if(power>0.f)return;
  const float alpha_cut=Mode==8?2.f/255.f:Mode==9?4.f/255.f:Mode==10?8.f/255.f:1.f/255.f;
  float alpha=min(.99f,co.w*(Mode==14?exp(power):Mode==26?boundedExp<1>(fmaxf(-80.f,power)):Mode==27?boundedExp<2>(fmaxf(-80.f,power)):__expf(power)));
  if(alpha<alpha_cut)return;
  const float next=T*(1-alpha);
  // 保留原F2在1e-4处的先验终止语义。额外近似提前退出在本成员正常累加之后。
  if(next<.0001f){done=true;return;}
  const float contribution=T*alpha;
  const float weight_cut=Mode==11?.0001f:Mode==12?.0005f:Mode==13?.001f:0.f;
  if(contribution>=weight_cut){
    C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  }
  T=next;
  const float stop=Mode==2?.0003f:Mode==3?.001f:Mode==4?.003f:Mode==5||Mode==14||Mode==24?.01f:Mode==25?.02f:Mode==6?.03f:Mode==7?.1f:0.f;
  if(stop>0.f&&T<stop)done=true;
}

template<int Mode> __device__ __forceinline__ void compositeApproxPair(
    float2 point,float4 co,float3 color,float2* pix,bool* done,float* T,float C[2][3],float bound) {
  #pragma unroll
  for(int p=0;p<2;++p)compositeApprox<Mode>(point,co,color,pix[p],done[p],T[p],C[p],bound);
}

template<int Mode> __global__ void renderApproxPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float* bg,float* out,const uint32_t* sorted_depth_keys,const GraphInputs* dynamic=nullptr) {
  if(dynamic){bg=dynamic->background;out=dynamic->image;}
  constexpr int Pixels=2,Rows=8,Threads=128;
  int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=threadIdx.y*blockDim.x+threadIdx.x;
  int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
  float T[Pixels],C[Pixels][3];float2 pix[Pixels];bool done[Pixels],inside[Pixels];
  #pragma unroll
  for(int p=0;p<Pixels;++p) {
    pix[p]={(float)x,(float)(y+p*Rows)};inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];
    T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;
  }
  if(sorted_depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=bg[ch];
    return;
  }
  int count=counts[tile];
  if(count<=capacity) {
    __shared__ float cached_bound[1];__shared__ float2 cached_xy[256];__shared__ float4 cached_conic[256];__shared__ float3 cached_color[256];
    for(int j=thread;j<count;j+=Threads) {
      int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_conic[j]=conic[id];
      cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    constexpr int Unroll=32;
    #pragma unroll Unroll
    for(int j=0;j<count;++j) {
      if((j&(Mode==23||Mode==24?31:7))==0) {
        bool all_done=true;
        #pragma unroll
        for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
        if(__all_sync(0xffffffff,all_done))break;
      }
      float2 point=cached_xy[j];float4 co=cached_conic[j];float3 color=cached_color[j];float bound=-CUDART_INF_F;
      compositeApproxPair<Mode>(point,co,color,pix,done,T,C,bound);
    }
  } else {
    for(int group=0;group<groups;++group) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty) {
        int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members) {
          int rank=w*32+__ffs(members)-1;members&=members-1;int id=sorted[rank];
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};float bound=-CUDART_INF_F;
          compositeApproxPair<Mode>(point,co,color,pix,done,T,C,bound);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}

