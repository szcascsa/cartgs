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

__device__ __forceinline__ float compensationInput(int point,int col,int64_t begin,int64_t n,
    const int64_t* ids,const int64_t* anchors,const float* offset,Tables tables,Controls control,
    const float* view,bool local,bool real){
  if(begin+point>=n)return 0.f;
  if(col<3)return offset[ids[point]*3+col];
  if(col<51){int64_t index=anchors[point]*48+col-3;float x=tables.p[0][index];
    for(int j=1;j<tables.count;++j){if(control.weights[j]<0.f)break;x=__fadd_rn(x,__fmul_rn(control.weights[j],tables.p[j][index]));}return x;}
  if(col<54)return control.ratio[col-51];if(col<61)return local&&real?view[anchors[point]*7+col-54]:0.f;
  return col==61&&local?1.f:0.f;
}
template<int Warps,bool Direct> __device__ __forceinline__ void compensationRegisterCore(int64_t n,
    const float* offset,const int64_t* rows,const int64_t* owners,const float* reference,Tables tables,
    Controls control,Weights w,const float* view,bool local,bool real,Master master,bool softplus,
    float* output,float* residual,int* status,const DeploymentFrame* frame,const DeploymentRequest* request,TensorWeights packed){
  if(status[0]==1)return;
  if(frame){n=frame->k;for(int j=0;j<6;++j)control.weights[j]=frame->weights[j];for(int j=0;j<3;++j)control.ratio[j]=frame->ratio_code[j];}
  constexpr int Points=Warps*16;int tid=threadIdx.x,warp=tid/32,lane=tid%32,group=lane/4,t=lane%4;
  int64_t begin=int64_t(blockIdx.x)*Points;if(begin>=n)return;
  __shared__ int64_t ids[Points],anchors[Points];
  __shared__ __align__(32) float delta[Points][20];
  __shared__ __align__(32) __half input[Points][72];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  for(int i=tid;i<Points*64;i+=blockDim.x){int p=i/64,c=i%64;input[p][c]=__float2half_rn(compensationInput(p,c,begin,n,ids,anchors,offset,tables,control,view,local,real));}
  __syncthreads();
  if(warp<Warps){int p0=warp*16+group,p1=p0+8;float first[4][4]{};unsigned a[4];
    #pragma unroll
    for(int k=0;k<64;k+=16){int j=k+t*2;
      a[0]=*reinterpret_cast<const unsigned*>(&input[p0][j]);
      a[1]=*reinterpret_cast<const unsigned*>(&input[p1][j]);
      a[2]=*reinterpret_cast<const unsigned*>(&input[p0][j+8]);
      a[3]=*reinterpret_cast<const unsigned*>(&input[p1][j+8]);
      #pragma unroll
      for(int ob=0;ob<4;++ob)registerProduct<32,64>(a,static_cast<const __half*>(packed.p[1]),k,ob,first[ob]);}
    float second[4][4]{};
    #pragma unroll
    for(int k=0;k<32;k+=16){activationA(first,w.p[1],k,a);
      #pragma unroll
      for(int ob=0;ob<4;++ob)registerProduct<32>(a,static_cast<const __half*>(packed.p[2]),k,ob,second[ob]);}
    float head[2][4]{};
    #pragma unroll
    for(int k=0;k<32;k+=16){activationA(second,w.p[3],k,a);
      #pragma unroll
      for(int ob=0;ob<2;++ob)registerProduct<16>(a,static_cast<const __half*>(packed.p[3]),k,ob,head[ob]);}
    #pragma unroll
    for(int ob=0;ob<2;++ob){
      #pragma unroll
      for(int q=0;q<4;++q){int p=q<2?p0:p1,c=ob*8+t*2+q%2;if(c>=14)continue;
        int h=c<3?0:c<6?1:c<10?2:3,first=h==0?0:h==1?3:h==2?6:10;
        float x=head[ob][q]+w.p[6+2*h][c-first];if(!h)x*=reference[ids[p]];delta[p][c]=x;
        if(begin+p<n){if(residual)residual[(begin+p)*14+c]=x;if(!isfinite(x))atomicExch(status,2);}
      }
    }
  }
  __syncthreads();int p=tid/8;
  if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],delta[p],master,softplus,output,status,tid%8,8,request);
  float* dest=Direct?request->attributes:output;
  for(int i=tid;i<Points*master.rest;i+=blockDim.x){int p=i/master.rest,c=i%master.rest;if(begin+p<n){float x=master.p[5][ids[p]*master.rest+c];dest[n*14+int64_t(begin+p)*master.rest+c]=x;if(!isfinite(x))atomicExch(status,2);}}
}
