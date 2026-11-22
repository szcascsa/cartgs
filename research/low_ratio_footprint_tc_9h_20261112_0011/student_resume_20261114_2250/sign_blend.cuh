// 用T符号表示像素已结束；幅值仍用于最终背景合成，逐成员顺序不变。
template<int Kind,int Step> __device__ __forceinline__ void signBlend(
    float power,float3 color,float& T,float* C){
  const float alpha=mergedFoldedAlpha(power,T<0.f);
  const float weight=alpha*T;
  const float next=Kind==1?T*(1.f-alpha):T-weight;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  T=__int_as_float(__float_as_int(next)|(next<.01f?int(0x80000000u):0));
}

template<int Kind,int Step> __device__ __forceinline__ void signPair(FoldedGaussian f,float3 color,
    float x,float y,float* T,float C[2][3]){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);
  signBlend<Kind,Step>(power,color,T[0],C[0]);signBlend<Kind,Step>(next,color,T[1],C[1]);
}

__device__ __forceinline__ void signFallback(float2 xy,float4 conic,float3 rgb,float2* pix,float* T,float C[2][3]){
  bool done[2]={T[0]<0.f,T[1]<0.f};T[0]=fabsf(T[0]);T[1]=fabsf(T[1]);
  compositeScalingPair<9>(xy,conic,rgb,pix,done,T,C,-CUDART_INF_F);
  if(done[0])T[0]=-T[0];if(done[1])T[1]=-T[1];
}
