// 独立冻结Teacher候选：第一层及输出保持FP32，五个64x64隐藏矩阵使用MMA。
// 四Head独立权重/偏置及14D物理解码均保留。近似只来自隐藏激活/权重量化与MMA累加。
template<int Mode> __global__ void packTeacherTensorWeights(Weights w,int which,
    typename TensorMath<Mode>::Scalar* output){
  const int index=blockIdx.x*blockDim.x+threadIdx.x;
  const int size=which==3?4*64*64:64*64;if(index>=size)return;
  float value=0.f;
  if(which==2)value=w.p[2][index];
  else if(which==3){const int head=index/(64*64);value=w.p[5+4*head][index%(64*64)];}
  output[index]=TensorMath<Mode>::convert(value);
}

template<int Mode,int Warps,bool Direct=false>
__global__ void compensationTeacherTensor(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,Weights w,
    const float* view,const float* view_weight,bool local,bool real_view,Master master,bool softplus,
    float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request,TensorWeights packed){
  if(status[0]==1)return;
  if(frame){n=frame->k;for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];}
  constexpr int Points=Warps*16,Threads=Warps*128,Accum=8;
  const int begin=blockIdx.x*Points;if(begin>=n)return;
  const int tid=threadIdx.x,warp=tid/32,col=tid%64,point=(tid/64)*Accum;
  using Math=TensorMath<Mode>;using Scalar=typename Math::Scalar;
  union SharedStage {float physical[Points][51];float values[Points][68];};
  __shared__ __align__(32) SharedStage scratch;
  float (*physical)[51]=scratch.physical;
  float (*values)[68]=scratch.values;
  __shared__ __align__(32) Scalar hidden[Points][64];
  __shared__ float delta[Points][14];
  __shared__ int64_t ids[Points],anchors[Points];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  for(int i=tid;i<Points*51;i+=Threads){
    const int p=i/51,c=i%51;float value=0.f;
    if(begin+p<n){
      if(c<3)value=offset[ids[p]*3+c];
      else{const auto index=anchors[p]*48+c-3;value=tables.p[0][index];
        for(int level=1;level<tables.count;++level){if(controls.weights[level]<0.f)break;
          value=__fadd_rn(value,__fmul_rn(controls.weights[level],tables.p[level][index]));}}
    }
    physical[p][c]=value;
  }
  __syncthreads();
  float accum[Accum]={};
  for(int j=0;j<51;++j){const float weight=w.p[0][j*64+col];
    #pragma unroll
    for(int p=0;p<Accum;++p)accum[p]=fmaf(physical[point+p][j],weight,accum[p]);}
  float ratio=0.f;for(int j=0;j<3;++j)ratio=fmaf(controls.ratio[j],w.p[4][j*64+col],ratio);
  #pragma unroll
  for(int p=0;p<Accum;++p){
    float value=accum[p]+w.p[1][col];value+=ratio;
    if(local){float projected=0.f;for(int j=0;j<7;++j)
        projected=fmaf(real_view?view[anchors[point+p]*7+j]:0.f,view_weight[j*64+col],projected);
      projected+=view_weight[7*64+col];value+=projected;}
    hidden[point+p][col]=Math::convert(relu(value));
  }
  __syncthreads();
  if(warp<Warps)tensorMultiply<Mode,64,64>(&hidden[warp*16][0],static_cast<const Scalar*>(packed.p[2]),&values[warp*16][0],68);
  __syncthreads();
  for(int i=tid;i<Points*64;i+=Threads)hidden[i/64][i%64]=Math::convert(relu(values[i/64][i%64]+w.p[3][i%64]));
  __syncthreads();
  for(int head=0;head<4;++head){
    const int weight=5+4*head,width=head<2?3:4,first=head==0?0:head==1?3:head==2?6:10;
    const Scalar* matrix=static_cast<const Scalar*>(packed.p[3])+head*64*64;
    if(warp<Warps)tensorMultiply<Mode,64,64>(&hidden[warp*16][0],matrix,&values[warp*16][0],68);
    __syncthreads();
    for(int i=tid;i<Points*64;i+=Threads)values[i/64][i%64]=relu(values[i/64][i%64]+w.p[weight+1][i%64]);
    __syncthreads();
    for(int i=tid;i<Points*width;i+=Threads){
      const int p=i/width,c=i%width;float value=0.f;
      for(int j=0;j<64;++j)value=fmaf(values[p][j],w.p[weight+2][j*width+c],value);
      value+=w.p[weight+3][c];if(!head&&begin+p<n)value*=reference[ids[p]];
      delta[p][first+c]=value;
      if(begin+p<n){if(residual)residual[(begin+p)*14+first+c]=value;if(!isfinite(value))atomicExch(status,2);}
    }
    __syncthreads();
  }
  const int p=tid/8;
  if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],delta[p],master,softplus,output,status,tid%8,8,request);
  float* destination=Direct?request->attributes:output;
  for(int i=tid;i<Points*master.rest;i+=Threads){const int p=i/master.rest,c=i%master.rest;
    if(begin+p<n){const float value=master.p[5][ids[p]*master.rest+c];
      destination[n*14+int64_t(begin+p)*master.rest+c]=value;if(!isfinite(value))atomicExch(status,2);}}
}
