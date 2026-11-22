// PTX m16n8k16布局：每线程持有2行各2列的FP32 accumulator。
// 上一层C直接打包为下一层A，隐藏激活不经过共享内存。
__device__ __forceinline__ unsigned halfPair(float a,float b){
  return unsigned(__half_as_ushort(__float2half_rn(a)))|(unsigned(__half_as_ushort(__float2half_rn(b)))<<16);
}
__device__ __forceinline__ unsigned halfPair(const __half* x,int a,int b){
  return unsigned(__half_as_ushort(x[a]))|(unsigned(__half_as_ushort(x[b]))<<16);
}
__device__ __forceinline__ void registerMma(const unsigned* a,unsigned b0,unsigned b1,float* c){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
template<int Columns,int Inner=32> __device__ __forceinline__ void registerProduct(const unsigned* a,
    const __half* weights,int k,int output_block,float* c){
  int lane=threadIdx.x%32,group=lane/4,t=lane%4,col=output_block*8+group,row=k+t*2;
  const unsigned* source=reinterpret_cast<const unsigned*>(weights+col*Inner+row);
  unsigned b0=source[0],b1=source[4];
  registerMma(a,b0,b1,c);
}
__device__ __forceinline__ void activationA(const float c[4][4],const float* bias,int k,unsigned* a){
  int t=threadIdx.x%4,j=k/8,col=k+t*2;
  float b0=bias[col],b1=bias[col+1],b8=bias[col+8],b9=bias[col+9];
  a[0]=halfPair(relu(c[j][0]+b0),relu(c[j][1]+b1));
  a[1]=halfPair(relu(c[j][2]+b0),relu(c[j][3]+b1));
  a[2]=halfPair(relu(c[j+1][0]+b8),relu(c[j+1][1]+b9));
  a[3]=halfPair(relu(c[j+1][2]+b8),relu(c[j+1][3]+b9));
}
template<int Width> __device__ __forceinline__ float selectorDynamic(int64_t row,int64_t n,int col,
    const int64_t* rows,const int64_t* owners,const float* view,float ratio,bool local,bool real){
  int j=col+10;if(row>=n||j>=Width)return 0.f;
  if(j==10)return ratio;if(j==18)return local?1.f:0.f;
  return local&&real?view[owners[rows[row]]*7+j-11]:0.f;
}
template<int Warps,int Width> __device__ __forceinline__ void selectorRegisterCore(int64_t n,
    const int64_t* rows,const int64_t* owners,const float* view,const float* noise,Weights w,float log_ratio,
    bool local,bool real,float* scores,float* logits,int* status,const DeploymentFrame* frame,
    const float* prefix,const void* body_weight,const void* dynamic_weight){
  if(frame){if(!frame->k)return;n=frame->n;log_ratio=frame->log_ratio;}
  int warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,t=lane%4;
  if(warp>=Warps)return;
  int64_t begin=(int64_t(blockIdx.x)*Warps+warp)*16,r0=begin+group,r1=r0+8;
  if(begin>=n)return;
  unsigned a[4];int j=t*2;
  a[0]=halfPair(selectorDynamic<Width>(r0,n,j,rows,owners,view,log_ratio,local,real),selectorDynamic<Width>(r0,n,j+1,rows,owners,view,log_ratio,local,real));
  a[1]=halfPair(selectorDynamic<Width>(r1,n,j,rows,owners,view,log_ratio,local,real),selectorDynamic<Width>(r1,n,j+1,rows,owners,view,log_ratio,local,real));
  a[2]=halfPair(selectorDynamic<Width>(r0,n,j+8,rows,owners,view,log_ratio,local,real),selectorDynamic<Width>(r0,n,j+9,rows,owners,view,log_ratio,local,real));
  a[3]=halfPair(selectorDynamic<Width>(r1,n,j+8,rows,owners,view,log_ratio,local,real),selectorDynamic<Width>(r1,n,j+9,rows,owners,view,log_ratio,local,real));
  float first[4][4]{};
  #pragma unroll
  for(int ob=0;ob<4;++ob){
    registerProduct<32,16>(a,static_cast<const __half*>(dynamic_weight),0,ob,first[ob]);
    int c=ob*8+t*2;int64_t id0=r0<n?rows[r0]:0,id1=r1<n?rows[r1]:0;
    first[ob][0]=(r0<n?prefix[id0*32+c]:0.f)+first[ob][0];
    first[ob][1]=(r0<n?prefix[id0*32+c+1]:0.f)+first[ob][1];
    first[ob][2]=(r1<n?prefix[id1*32+c]:0.f)+first[ob][2];
    first[ob][3]=(r1<n?prefix[id1*32+c+1]:0.f)+first[ob][3];
  }
  float second[4][4]{};
  #pragma unroll
  for(int k=0;k<32;k+=16){activationA(first,w.p[1],k,a);
    #pragma unroll
    for(int ob=0;ob<4;++ob)registerProduct<32>(a,static_cast<const __half*>(body_weight),k,ob,second[ob]);}
  float head[4]{};
  #pragma unroll
  for(int k=0;k<32;k+=16){activationA(second,w.p[3],k,a);
    int kr=k+t*2;unsigned b0=0,b1=0;
    if(group<2){b0=halfPair(w.p[4][kr*2+group],w.p[4][(kr+1)*2+group]);b1=halfPair(w.p[4][(kr+8)*2+group],w.p[4][(kr+9)*2+group]);}
    registerMma(a,b0,b1,head);}
  if(t==0){
    #pragma unroll
    for(int q=0;q<2;++q){int64_t row=q?r1:r0;if(row>=n)continue;
      float x=head[q*2]+w.p[5][0],y=head[q*2+1]+w.p[5][1];
      if(logits){logits[row*2]=x;logits[row*2+1]=y;}
      if(!isfinite(x)||!isfinite(y))atomicExch(status,1);
      int64_t id=rows[row];x+=noise[id*2];y+=noise[id*2+1];float mx=fmaxf(x,y),ea=expf(x-mx),eb=expf(y-mx);
      scores[row]=eb/(ea+eb);if(!isfinite(scores[row]))atomicExch(status,1);
    }
  }
}

