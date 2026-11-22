template<int Mode,int Warps,bool Direct=false,bool Projected=false>
__global__ void compensationStudentScheduled(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,Weights w,
    const float* view,const float* view_weight,bool local,bool real_view,Master master,
    bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request,TensorWeights packed){
  if(status[0]==1)return;
  constexpr int Points=Warps*16;
  int tid=threadIdx.x,warp=tid/32,first_begin=blockIdx.x*Points;
  if(frame)n=frame->k;
  if(first_begin>=n)return;
  if(frame){for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];}
  using Math=TensorMath<Mode>;using Scalar=typename Math::Scalar;
  __shared__ __align__(32) Scalar inputs[Points][Projected?32:64];
  __shared__ __align__(32) float values[Points][36];
  Scalar* hidden=&inputs[0][0];
  __shared__ int64_t ids[Points],anchors[Points];
  for(int64_t begin=first_begin;begin<n;begin+=int64_t(gridDim.x)*Points){
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
  if constexpr(Projected){
    const float* fixed=static_cast<const float*>(packed.p[5]);
    const float* dynamic=static_cast<const float*>(packed.p[7]);
    for(int i=tid;i<Points*32;i+=blockDim.x){int p=i/32,c=i%32;float x=0.f;
      if(begin+p<n)x=__fadd_rn(fixed[ids[p]*32+c],dynamic[anchors[p]*32+c]);
      hidden[i]=Math::convert(relu(__fadd_rn(x,w.p[1][c])));}
  }else{
  for(int i=tid;i<Points*64;i+=blockDim.x){
    int p=i/64,c=i%64;float x=0.f;
    if(begin+p<n){if(c<3)x=offset[ids[p]*3+c];else if(c<51){int64_t index=anchors[p]*48+c-3;x=tables.p[0][index];
      for(int j=1;j<tables.count;++j){if(controls.weights[j]<0.f)break;x=__fadd_rn(x,__fmul_rn(controls.weights[j],tables.p[j][index]));}}
      else if(c<54)x=controls.ratio[c-51];
      else if(c<61)x=local&&real_view?view[anchors[p]*7+c-54]:0.f;
      else if(c==61)x=local?1.f:0.f;}
    inputs[p][c]=Math::convert(x);
  }
  __syncthreads();
  if(warp<Warps)tensorMultiply<Mode,64,32>(&inputs[warp*16][0],static_cast<const Scalar*>(packed.p[1]),&values[warp*16][0],36);
  __syncthreads();
  for(int i=tid;i<Points*32;i+=blockDim.x)hidden[i]=Math::convert(relu(values[i/32][i%32]+w.p[1][i%32]));
  }
  __syncthreads();
  if(warp<Warps)tensorMultiply<Mode,32,32>(hidden+warp*16*32,static_cast<const Scalar*>(packed.p[2]),&values[warp*16][0],36);
  __syncthreads();
  for(int i=tid;i<Points*32;i+=blockDim.x)hidden[i]=Math::convert(relu(values[i/32][i%32]+w.p[3][i%32]));
  __syncthreads();
  if(warp<Warps)tensorMultiply<Mode,32,16>(hidden+warp*16*32,static_cast<const Scalar*>(packed.p[3]),&values[warp*16][0],36);
  __syncthreads();
  for(int i=tid;i<Points*14;i+=blockDim.x){int p=i/14,c=i%14,head=c<3?0:c<6?1:c<10?2:3;
    int first=head==0?0:head==1?3:head==2?6:10;float value=values[p][c]+w.p[6+2*head][c-first];
    if(!head)value*=reference[ids[p]];values[p][c]=value;
    if(begin+p<n){if(residual)residual[(begin+p)*14+c]=value;if(!isfinite(value))atomicExch(status,2);}}
  __syncthreads();
  // 与原实现共用物理发布函数：FP32 Scale/Quaternion/Opacity及SH、ID顺序不变。
  for(int p=tid/8;p<Points;p+=blockDim.x/8)
    if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],values[p],master,softplus,output,status,tid%8,8,request);
  float* destination=Direct?request->attributes:output;
  for(int i=tid;i<Points*master.rest;i+=blockDim.x){int p=i/master.rest,c=i%master.rest;
    if(begin+p<n){float x=master.p[5][ids[p]*master.rest+c];destination[n*14+int64_t(begin+p)*master.rest+c]=x;if(!isfinite(x))atomicExch(status,2);}}
  __syncthreads();
  }
}
