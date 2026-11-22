// 仅冻结推理。MMA乘法采用TF32或FP16，累加与物理解码仍为FP32。
struct TensorWeights { const void* p[5]; };
template<int Mode> struct TensorMath;
template<> struct TensorMath<1> {
  using Scalar=float;
  using Operand=nvcuda::wmma::precision::tf32;
  static constexpr int Inner=8;
  __device__ static float convert(float x){return nvcuda::wmma::__float_to_tf32(x);}
};
template<> struct TensorMath<2> {
  using Scalar=__half;
  using Operand=__half;
  static constexpr int Inner=16;
  __device__ static __half convert(float x){return __float2half_rn(x);}
};

template<int Mode,int Inner,int Columns>
__device__ void tensorMultiply(const typename TensorMath<Mode>::Scalar* a,
    const typename TensorMath<Mode>::Scalar* b,float* out,int output_stride){
  namespace wm=nvcuda::wmma;
  using Kind=typename TensorMath<Mode>::Operand;
  constexpr int Step=TensorMath<Mode>::Inner;
  wm::fragment<wm::accumulator,16,16,Step,float> c[Columns/16];
  #pragma unroll
  for(int j=0;j<Columns/16;++j)wm::fill_fragment(c[j],0.f);
  #pragma unroll
  for(int k=0;k<Inner;k+=Step){
    wm::fragment<wm::matrix_a,16,16,Step,Kind,wm::row_major> x;
    wm::load_matrix_sync(x,a+k,Inner);
    #pragma unroll
    for(int j=0;j<Columns/16;++j){
      wm::fragment<wm::matrix_b,16,16,Step,Kind,wm::row_major> w;
      wm::load_matrix_sync(w,b+k*Columns+j*16,Columns);
      wm::mma_sync(c[j],x,w,c[j]);
    }
  }
  #pragma unroll
  for(int j=0;j<Columns/16;++j)wm::store_matrix_sync(out+j*16,c[j],output_stride,wm::mem_row_major);
}

template<int Mode> __global__ void packTensorWeights(Weights selector,Weights student,
    const float* view_weight,int selector_width,int which,typename TensorMath<Mode>::Scalar* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int size=which==1?64*32:which==3||which==4?32*16:32*32;
  if(i>=size)return;
  float x=0.f;
  if(which==0)x=selector.p[2][i];
  else if(which==1){int row=i/32,col=i%32;
    if(row<51)x=student.p[0][i];
    else if(row<54)x=student.p[4][(row-51)*32+col];
    else if(row<62&&view_weight)x=view_weight[(row-54)*32+col];}
  else if(which==2)x=student.p[2][i];
  else if(which==3){
    int row=i/16,c=i%16,head=c<3?0:c<6?1:c<10?2:3;
    int first=head==0?0:head==1?3:head==2?6:10,width=head<2?3:4;
    if(c<14)x=student.p[5+2*head][row*width+c-first];
  }else if(i/32+10<selector_width)x=selector.p[0][(i/32+10)*32+i%32];
  out[i]=TensorMath<Mode>::convert(x);
}

#include "tensor_register.cuh"

template<int Mode,int Warps,int Width>
__global__ void selectorTensor(int64_t n,const float* physical,const int64_t* rows,
    const int64_t* owners,const float* view,const float* noise,Weights w,float log_ratio,
    bool local,bool real_view,float* scores,float* logits,int* status,
    const DeploymentFrame* frame,const float* prefix,const void* packed_weight,const void* dynamic_weight){
  if constexpr(Mode==2){if(prefix){selectorRegisterCore<Warps,Width>(n,rows,owners,view,noise,w,log_ratio,local,real_view,scores,logits,status,frame,prefix,packed_weight,dynamic_weight);return;}}
  if(frame){if(!frame->k)return;n=frame->n;log_ratio=frame->log_ratio;}
  constexpr int Points=Warps*16;
  int tid=threadIdx.x,warp=tid/32,begin=blockIdx.x*Points;
  if(begin>=n)return;
  using Math=TensorMath<Mode>;using Scalar=typename Math::Scalar;
  __shared__ __align__(32) Scalar h0[Points][32];
  __shared__ __align__(32) float h1[Points][36];
  __shared__ float result[Points][2];
  if(prefix){
    Scalar* input=&h0[0][0];
    for(int i=tid;i<Points*16;i+=blockDim.x){int p=i/16,c=i%16,j=c+10;float x=0.f;
      if(begin+p<n&&j<Width){int64_t id=rows[begin+p];x=j==10?log_ratio:j<18?(local&&real_view?view[owners[id]*7+j-11]:0.f):(local?1.f:0.f);}
      input[i]=Math::convert(x);}
    __syncthreads();
    if(warp<Warps)tensorMultiply<Mode,16,32>(input+warp*16*16,static_cast<const Scalar*>(dynamic_weight),&h1[warp*16][0],36);
    __syncthreads();
    for(int i=tid;i<Points*32;i+=blockDim.x){int p=i/32,c=i%32;float x=begin+p<n?prefix[rows[begin+p]*32+c]:0.f;
      h0[p][c]=Math::convert(relu((x+h1[p][c])+w.p[1][c]));}
  }else for(int i=tid;i<Points*32;i+=blockDim.x){
    int p=i/32,col=i%32;float sum=0.f;
    if(begin+p<n){
      int64_t id=rows[begin+p];
      if(prefix)sum=prefix[id*32+col];
      for(int j=prefix?10:0;j<Width;++j){
        float x=j<10?physical[id*10+j]:j==10?log_ratio:
            j<18?(local&&real_view?view[owners[id]*7+j-11]:0.f):(local?1.f:0.f);
        sum=fmaf(x,w.p[0][j*32+col],sum);
      }
      sum=relu(sum+w.p[1][col]);
    }
    h0[p][col]=Math::convert(sum);
  }
  __syncthreads();
  if(warp<Warps)tensorMultiply<Mode,32,32>(&h0[warp*16][0],static_cast<const Scalar*>(packed_weight),&h1[warp*16][0],36);
  __syncthreads();
  for(int i=tid;i<Points*32;i+=blockDim.x)h1[i/32][i%32]=relu(h1[i/32][i%32]+w.p[3][i%32]);
  __syncthreads();
  // 2列输出与概率仍FP32；隐藏层微差可能改变分数排序，必须另行量化。
  int point=tid/2,column=tid%2;
  if(point<Points && begin+point<n){float sum=0.f;for(int j=0;j<32;++j)sum=fmaf(h1[point][j],w.p[4][j*2+column],sum);
    result[point][column]=sum+w.p[5][column];if(logits)logits[(begin+point)*2+column]=result[point][column];if(!isfinite(result[point][column]))atomicExch(status,1);}
  __syncthreads();
  for(int p=tid;p<Points;p+=blockDim.x)if(begin+p<n){
    int64_t id=rows[begin+p];float a=result[p][0]+noise[id*2],b=result[p][1]+noise[id*2+1];
    float mx=fmaxf(a,b),ea=expf(a-mx),eb=expf(b-mx);scores[begin+p]=eb/(ea+eb);
    if(!isfinite(scores[begin+p]))atomicExch(status,1);
  }
}

template<int Mode,int Warps,bool Direct=false>
__global__ void compensationTensor(int64_t n,const float* offset,const int64_t* rows,
    const int64_t* owners,const float* reference,Tables tables,Controls controls,Weights w,
    const float* view,const float* view_weight,bool local,bool real_view,Master master,
    bool softplus,float* output,float* residual,int* status,const DeploymentFrame* frame,
    const DeploymentRequest* request,TensorWeights packed){
  if constexpr(Mode==2){compensationRegisterCore<Warps,Direct>(n,offset,rows,owners,reference,tables,controls,w,view,local,real_view,master,softplus,output,residual,status,frame,request,packed);return;}
  if(status[0]==1)return;
  if(frame){n=frame->k;for(int j=0;j<6;++j)controls.weights[j]=frame->weights[j];
    for(int j=0;j<3;++j)controls.ratio[j]=frame->ratio_code[j];}
  constexpr int Points=Warps*16;
  int tid=threadIdx.x,warp=tid/32,begin=blockIdx.x*Points;
  if(begin>=n)return;
  using Math=TensorMath<Mode>;using Scalar=typename Math::Scalar;
  __shared__ __align__(32) Scalar inputs[Points][64];
  __shared__ __align__(32) float values[Points][36];
  Scalar* hidden=&inputs[0][0];
  __shared__ int64_t ids[Points],anchors[Points];
  if(tid<Points){ids[tid]=begin+tid<n?rows[begin+tid]:0;anchors[tid]=begin+tid<n?owners[ids[tid]]:0;}
  __syncthreads();
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
  const int p=tid/8;
  if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],values[p],master,softplus,output,status,tid%8,8,request);
  float* destination=Direct?request->attributes:output;
  for(int i=tid;i<Points*master.rest;i+=blockDim.x){int p=i/master.rest,c=i%master.rest;
    if(begin+p<n){float x=master.p[5][ids[p]*master.rest+c];destination[n*14+int64_t(begin+p)*master.rest+c]=x;if(!isfinite(x))atomicExch(status,2);}}
}
