// Student 32D隐藏层直接将MMA accumulator转成下一层A寄存器。
// 各warp处理16个Gaussian，四个Head按原权重拼入14个独立输出列。
__global__ void packStudentRegisterWeights(Weights weights,int which,__half* output){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int columns=which==2?32:16;
  if(i>=32*columns)return;
  const int row=i%32,col=i/32;
  float value=0.f;
  if(which==2)value=weights.p[2][row*32+col];
  else if(col<14){
    const int head=col<3?0:col<6?1:col<10?2:3;
    const int first=head==0?0:head==1?3:head==2?6:10,width=head<2?3:4;
    value=weights.p[5+2*head][row*width+col-first];
  }
  output[i]=__float2half_rn(value);
}

__device__ __forceinline__ float studentProjectedActivation(int64_t row,int64_t n,int column,
    const int64_t* rows,const int64_t* owners,const float* fixed,const float* dynamic,const float* bias){
  if(row>=n)return 0.f;
  const int64_t id=rows[row],anchor=owners[id];
  return relu(__fadd_rn(__fadd_rn(fixed[id*32+column],dynamic[anchor*32+column]),bias[column]));
}

template<int Warps,bool Direct=false>
__global__ void compensationStudentRegisters(int64_t capacity,const int64_t* rows,const int64_t* owners,
    const float* reference,Weights weights,Master master,bool softplus,float* output,int* status,
    const DeploymentFrame* frame,const DeploymentRequest* request,TensorWeights packed){
  if(status[0]==1)return;
  const int64_t n=frame?frame->k:capacity;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,group=lane/4,t=lane%4;
  constexpr int Points=Warps*16;
  __shared__ float residuals[Points][14];
  const float* fixed=static_cast<const float*>(packed.p[5]);
  const float* dynamic=static_cast<const float*>(packed.p[7]);
  for(int64_t block_begin=int64_t(blockIdx.x)*Points;block_begin<n;block_begin+=int64_t(gridDim.x)*Points){
    const int64_t begin=block_begin+warp*16,r0=begin+group,r1=r0+8;
    float hidden[4][4]{};
    if(begin<n){
      #pragma unroll
      for(int k=0;k<32;k+=16){
        const int c=k+t*2;
        unsigned a[4]={
          halfPair(studentProjectedActivation(r0,n,c,rows,owners,fixed,dynamic,weights.p[1]),studentProjectedActivation(r0,n,c+1,rows,owners,fixed,dynamic,weights.p[1])),
          halfPair(studentProjectedActivation(r1,n,c,rows,owners,fixed,dynamic,weights.p[1]),studentProjectedActivation(r1,n,c+1,rows,owners,fixed,dynamic,weights.p[1])),
          halfPair(studentProjectedActivation(r0,n,c+8,rows,owners,fixed,dynamic,weights.p[1]),studentProjectedActivation(r0,n,c+9,rows,owners,fixed,dynamic,weights.p[1])),
          halfPair(studentProjectedActivation(r1,n,c+8,rows,owners,fixed,dynamic,weights.p[1]),studentProjectedActivation(r1,n,c+9,rows,owners,fixed,dynamic,weights.p[1]))};
        #pragma unroll
        for(int ob=0;ob<4;++ob)registerProduct<32>(a,static_cast<const __half*>(packed.p[2]),k,ob,hidden[ob]);
      }
      float heads[2][4]{};
      #pragma unroll
      for(int k=0;k<32;k+=16){
        unsigned a[4];activationA(hidden,weights.p[3],k,a);
        #pragma unroll
        for(int ob=0;ob<2;++ob)registerProduct<16>(a,static_cast<const __half*>(packed.p[3]),k,ob,heads[ob]);
      }
      #pragma unroll
      for(int ob=0;ob<2;++ob){
        #pragma unroll
        for(int q=0;q<4;++q){
          const int column=ob*8+t*2+(q%2),point=warp*16+group+(q/2)*8;
          const int64_t row=block_begin+point;
          if(column<14){
            const int head=column<3?0:column<6?1:column<10?2:3;
            const int first=head==0?0:head==1?3:head==2?6:10;
            float value=heads[ob][q]+weights.p[6+2*head][column-first];
            if(!head&&row<n)value*=reference[rows[row]];
            residuals[point][column]=value;
            if(row<n&&!isfinite(value))atomicExch(status,2);
          }
        }
      }
    }
    __syncthreads();
    for(int p=tid/8;p<Points;p+=blockDim.x/8){
      const int64_t row=block_begin+p;
      if(row<n)publishPoint<Direct,false>(row,n,rows[row],residuals[p],master,softplus,output,status,tid%8,8,request);
    }
    float* destination=Direct?request->attributes:output;
    for(int i=tid;i<Points*master.rest;i+=blockDim.x){
      const int point=i/master.rest,c=i%master.rest;const int64_t row=block_begin+point;
      if(row<n){const float value=master.p[5][rows[row]*master.rest+c];
        destination[n*14+row*master.rest+c]=value;if(!isfinite(value))atomicExch(status,2);}
    }
    __syncthreads();
  }
}
