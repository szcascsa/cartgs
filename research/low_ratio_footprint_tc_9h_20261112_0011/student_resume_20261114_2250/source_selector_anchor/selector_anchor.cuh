template<int Width> __device__ __forceinline__ float selectorAnchorValue(int64_t row,int64_t n,int col,const float* view,float ratio,bool local,bool real){
  const int j=col+10;if(row>=n||j>=Width)return 0.f;
  if(j==10)return ratio;if(j==18)return local?1.f:0.f;
  return local&&real?view[row*7+j-11]:0.f;
}
template<int Width> __global__ void selectorAnchorProjection(int64_t n,const float* view,const DeploymentFrame* frame,bool local,bool real,const void* weight,float* output){
  if(!frame->k)return;
  const int lane=threadIdx.x%32,group=lane/4,t=lane%4;
  const int64_t begin=(int64_t(blockIdx.x)*4+threadIdx.x/32)*16,r0=begin+group,r1=r0+8;
  if(begin>=n)return;
  const float ratio=frame->log_ratio;unsigned a[4];const int j=t*2;
  a[0]=halfPair(selectorAnchorValue<Width>(r0,n,j,view,ratio,local,real),selectorAnchorValue<Width>(r0,n,j+1,view,ratio,local,real));
  a[1]=halfPair(selectorAnchorValue<Width>(r1,n,j,view,ratio,local,real),selectorAnchorValue<Width>(r1,n,j+1,view,ratio,local,real));
  a[2]=halfPair(selectorAnchorValue<Width>(r0,n,j+8,view,ratio,local,real),selectorAnchorValue<Width>(r0,n,j+9,view,ratio,local,real));
  a[3]=halfPair(selectorAnchorValue<Width>(r1,n,j+8,view,ratio,local,real),selectorAnchorValue<Width>(r1,n,j+9,view,ratio,local,real));
  #pragma unroll
  for(int ob=0;ob<4;++ob){
    float c[4]{};registerProduct<32,16>(a,static_cast<const __half*>(weight),0,ob,c);
    const int col=ob*8+t*2;
    if(r0<n){output[r0*32+col]=c[0];output[r0*32+col+1]=c[1];}
    if(r1<n){output[r1*32+col]=c[2];output[r1*32+col+1]=c[3];}
  }
}
template<int Warps,int Width> __global__ void selectorProjected(int64_t n,
    const int64_t* rows,const int64_t* owners,const float* view,const float* noise,Weights w,float log_ratio,
    bool local,bool real,float* scores,float* logits,int* status,const DeploymentFrame* frame,
    const float* prefix,const void* body_weight,const float* projected){
  if(frame){if(!frame->k)return;n=frame->n;log_ratio=frame->log_ratio;}
  int warp=threadIdx.x/32,lane=threadIdx.x%32,group=lane/4,t=lane%4;
  if(warp>=Warps)return;
  int64_t begin=(int64_t(blockIdx.x)*Warps+warp)*16,r0=begin+group,r1=r0+8;
  if(begin>=n)return;
  unsigned a[4];
  float first[4][4]{};
  #pragma unroll
  for(int ob=0;ob<4;++ob){

    int c=ob*8+t*2;int64_t id0=r0<n?rows[r0]:0,id1=r1<n?rows[r1]:0;
    first[ob][0]=(r0<n?prefix[id0*32+c]:0.f)+(r0<n?projected[owners[id0]*32+c]:0.f);
    first[ob][1]=(r0<n?prefix[id0*32+c+1]:0.f)+(r0<n?projected[owners[id0]*32+c+1]:0.f);
    first[ob][2]=(r1<n?prefix[id1*32+c]:0.f)+(r1<n?projected[owners[id1]*32+c]:0.f);
    first[ob][3]=(r1<n?prefix[id1*32+c+1]:0.f)+(r1<n?projected[owners[id1]*32+c+1]:0.f);
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

