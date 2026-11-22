// 冻结64D投影：只缓存地图/Feature/权重相关项；当前Ratio、View每帧重算。
__global__ void preparationProjectTeacherAnchors(int64_t n,const float* residual,int levels,
    Controls controls,Weights weights,const float* view,const float* view_weight,bool local,bool real_view,
    float* output,const DeploymentFrame* frame,const bool* visible){
  if(frame&&!frame->k)return;
  const int c=threadIdx.x%64;const int64_t anchor=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/64;
  if(anchor>=n)return;
  if(visible&&!visible[anchor]){if(!anchor)output[c]=0.f;return;}
  if(frame){for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];}
  float value=0.f;
  for(int j=0;j<levels;++j){const float w=controls.weights[j+1];if(w<0.f)break;
    value=__fadd_rn(value,__fmul_rn(w,residual[(int64_t(j)*n+anchor)*64+c]));}
  for(int j=0;j<3;++j)value=__fmaf_rn(controls.ratio[j],weights.p[4][j*64+c],value);
  if(local)for(int j=0;j<8;++j)value=__fmaf_rn(j==7?1.f:(real_view?view[anchor*7+j]:0.f),view_weight[j*64+c],value);
  output[anchor*64+c]=value;
}

// 每个warp负责一个独立Head；原Head参数、ReLU和FP32最终点积保持独立。
template<int Mode,bool Direct=false,bool Projected=false>
__global__ void compensationTeacherParallel(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,Weights weights,
    const float* view,const float* view_weight,bool local,bool real_view,Master master,bool softplus,
    float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request,TensorWeights packed){
  if(status[0]==1)return;
  if(frame){n=frame->k;for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];}
  constexpr int Points=16,Threads=128,Accum=8;
  const int64_t first_begin=int64_t(blockIdx.x)*Points;if(first_begin>=n)return;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,col=tid%64,point=(tid/64)*Accum;
  using Math=TensorMath<Mode>;using Scalar=typename Math::Scalar;
  union SharedStage {float physical[Points][51];float values[4][Points][68];};
  __shared__ __align__(32) SharedStage scratch;
  __shared__ __align__(32) Scalar hidden[Points][64];
  __shared__ float delta[Points][14];__shared__ int64_t ids[Points],anchors[Points];
  for(int64_t begin=first_begin;begin<n;begin+=int64_t(gridDim.x)*Points){
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  if constexpr(Projected){
    const float* fixed=static_cast<const float*>(packed.p[5]);const float* dynamic=static_cast<const float*>(packed.p[7]);
    for(int i=tid;i<Points*64;i+=Threads){const int p=i/64,c=i%64;float value=0.f;
      if(begin+p<n)value=__fadd_rn(fixed[ids[p]*64+c],dynamic[anchors[p]*64+c]);
      hidden[p][c]=Math::convert(relu(__fadd_rn(value,weights.p[1][c])));}
  }else{
    for(int i=tid;i<Points*51;i+=Threads){const int p=i/51,c=i%51;float value=0.f;
      if(begin+p<n){if(c<3)value=offset[ids[p]*3+c];else{const int64_t index=anchors[p]*48+c-3;value=tables.p[0][index];
          for(int level=1;level<tables.count;++level){if(controls.weights[level]<0.f)break;
            value=__fadd_rn(value,__fmul_rn(controls.weights[level],tables.p[level][index]));}}}
      scratch.physical[p][c]=value;}
    __syncthreads();float accum[Accum]={};
    for(int j=0;j<51;++j){const float weight=weights.p[0][j*64+col];
      #pragma unroll
      for(int p=0;p<Accum;++p)accum[p]=fmaf(scratch.physical[point+p][j],weight,accum[p]);}
    float ratio=0.f;for(int j=0;j<3;++j)ratio=fmaf(controls.ratio[j],weights.p[4][j*64+col],ratio);
    #pragma unroll
    for(int p=0;p<Accum;++p){float value=accum[p]+weights.p[1][col];value+=ratio;
      if(local){float projected=0.f;for(int j=0;j<7;++j)projected=fmaf(real_view?view[anchors[point+p]*7+j]:0.f,view_weight[j*64+col],projected);
        projected+=view_weight[7*64+col];value+=projected;}
      hidden[point+p][col]=Math::convert(relu(value));}
  }
  __syncthreads();
  if(!warp)tensorMultiply<Mode,64,64>(&hidden[0][0],static_cast<const Scalar*>(packed.p[2]),&scratch.values[0][0][0],68);
  __syncthreads();
  for(int i=tid;i<Points*64;i+=Threads)hidden[i/64][i%64]=Math::convert(relu(scratch.values[0][i/64][i%64]+weights.p[3][i%64]));
  __syncthreads();
  const int head=warp,w=5+4*head,width=head<2?3:4,first=head==0?0:head==1?3:head==2?6:10;
  tensorMultiply<Mode,64,64>(&hidden[0][0],static_cast<const Scalar*>(packed.p[3])+head*64*64,&scratch.values[head][0][0],68);
  __syncwarp();
  for(int i=lane;i<Points*64;i+=32)scratch.values[head][i/64][i%64]=relu(scratch.values[head][i/64][i%64]+weights.p[w+1][i%64]);
  __syncwarp();
  for(int i=lane;i<Points*width;i+=32){const int p=i/width,c=i%width;float value=0.f;
    for(int j=0;j<64;++j)value=fmaf(scratch.values[head][p][j],weights.p[w+2][j*width+c],value);
    value+=weights.p[w+3][c];if(!head&&begin+p<n)value*=reference[ids[p]];delta[p][first+c]=value;
    if(begin+p<n){if(residual)residual[(begin+p)*14+first+c]=value;if(!isfinite(value))atomicExch(status,2);}}
  __syncthreads();
  const int p=tid/8;if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],delta[p],master,softplus,output,status,tid%8,8,request);
  float* destination=Direct?request->attributes:output;
  for(int i=tid;i<Points*master.rest;i+=Threads){const int p=i/master.rest,c=i%master.rest;
    if(begin+p<n){const float value=master.p[5][ids[p]*master.rest+c];destination[n*14+int64_t(begin+p)*master.rest+c]=value;
      if(!isfinite(value))atomicExch(status,2);}}
  __syncthreads();
  }
}
