// 一warp处理一Anchor：8个lane并行投影8个角点，当前View在寄存器内进入32D投影。
// View本身仍发布给Selector；不缓存任何当前视角结果。
template<int Lanes> __global__ void studentGroupedViewProject(int64_t n,const float* centers,const float* centers4,
    const float* corners,const float* camera,const float* vm,int64_t v0,int64_t v1,
    const float* pm,int64_t p0,int64_t p1,const float* scale,float aspect,float near,float* out,
    const DeploymentFrame* frame,const DeploymentRequest* request,const bool* visible,
    const float* residual,int levels,Weights weights,const float* view_weight,bool real_view,float* projected){
  if(!frame->k)return;
  const int lane=threadIdx.x%Lanes;
  const unsigned group_mask=(0xffffffffu>>(32-Lanes))<<((threadIdx.x%32)/Lanes*Lanes);
  const int64_t a=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/Lanes;
  if(a>=n)return;
  if(visible&&!visible[a]){
    if(!a){for(int col=lane;col<32;col+=Lanes)projected[col]=0.f;if(lane<7)out[lane]=0.f;}
    return;
  }
  if(request){aspect=request->aspect;near=request->near;}
  float lo0=CUDART_INF_F,lo1=CUDART_INF_F,hi0=-CUDART_INF_F,hi1=-CUDART_INF_F;
  float mind=CUDART_INF_F,maxd=-CUDART_INF_F;
  if(lane<8){
    const float* point=corners+a*32+lane*4;
    const float depth=dot4(point,vm,2,v0,v1),w=safeW(dot4(point,pm,3,p0,p1),near);
    lo0=hi0=dot4(point,pm,0,p0,p1)/w;
    lo1=hi1=dot4(point,pm,1,p0,p1)/w;
    mind=maxd=depth;
  }
  #pragma unroll
  for(int delta=4;delta;delta/=2){
    lo0=fminf(lo0,__shfl_xor_sync(group_mask,lo0,delta,Lanes));hi0=fmaxf(hi0,__shfl_xor_sync(group_mask,hi0,delta,Lanes));
    lo1=fminf(lo1,__shfl_xor_sync(group_mask,lo1,delta,Lanes));hi1=fmaxf(hi1,__shfl_xor_sync(group_mask,hi1,delta,Lanes));
    mind=fminf(mind,__shfl_xor_sync(group_mask,mind,delta,Lanes));maxd=fmaxf(maxd,__shfl_xor_sync(group_mask,maxd,delta,Lanes));
  }
  float value=0.f;
  if(lane<3){
    float v[3];for(int j=0;j<3;++j)v[j]=camera[j]-centers[a*3+j];
    const float norm=fmaxf(sqrtf((v[0]*v[0]+v[2]*v[2])+v[1]*v[1]),1e-6f);
    value=v[lane]/norm;
  }else if(lane==3)value=asinhf(dot4(centers4+a*4,vm,2,v0,v1)/scale[0]);
  else if(lane<6){
    const float w=safeW(dot4(centers4+a*4,pm,3,p0,p1),near);
    value=clamp(dot4(centers4+a*4,pm,lane-4,p0,p1)/w,-4.f,4.f);
  }else if(lane==6){
    float longest=clamp(fmaxf((hi0-lo0)*aspect*.5f,(hi1-lo1)*.5f),0.f,1e4f);
    if(mind<=near)longest=1e4f;
    if(maxd<=near)longest=0.f;
    value=log1pf(longest);
  }
  if(lane<7)out[a*7+lane]=value;
  float encoded[8];
  #pragma unroll
  for(int j=0;j<8;++j)encoded[j]=j==7?1.f:(real_view?__shfl_sync(group_mask,value,j,Lanes):0.f);
  #pragma unroll
  for(int part=0;part<32/Lanes;++part){
    const int col=lane+part*Lanes;float sum=0.f;
    for(int j=0;j<levels;++j){const float weight=frame->weights[j+1];if(weight<0.f)break;
      sum=__fadd_rn(sum,__fmul_rn(weight,residual[(int64_t(j)*n+a)*32+col]));}
    for(int j=0;j<3;++j)sum=__fmaf_rn(frame->ratio_code[j],weights.p[4][j*32+col],sum);
    #pragma unroll
    for(int j=0;j<8;++j)sum=__fmaf_rn(encoded[j],view_weight[j*32+col],sum);
    projected[a*32+col]=sum;
  }
}
