// 将bit-exp的线性变换提升到每Gaussian/Tile装载阶段；颜色和T仍为FP32。
__device__ __forceinline__ float encodedAlpha(float encoded,bool done){
  if(done)return 0.f;
  const int bits=min(__float2int_rz(encoded),__float_as_int(.99f));
  return bits<__float_as_int(1.f/255.f)?0.f:__int_as_float(bits);
}

template<int Kind,int Step> __device__ __forceinline__ void encodedBlend(
    float power,float3 color,bool& done,float& T,float* C){
  const float alpha=Kind==4?mergedFoldedAlpha(power,done):encodedAlpha(power,done);
  const float weight=alpha*T;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  T=T-weight;if(T<.01f)done=true;
}

template<int Kind,int Step> __device__ __forceinline__ void encodedPair(FoldedGaussian f,float3 color,
    float x,float y,bool* done,float* T,float C[2][3]){
  float power;
  if constexpr(Kind==1)power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  else power=fmaf(x,fmaf(f.a.x,x,fmaf(f.a.y,y,f.a.w)),fmaf(y,fmaf(f.a.z,y,f.b.x),f.b.y));
  float next;
  if constexpr(Kind==3){const float v=y+float(Step);next=fmaf(x,fmaf(f.a.x,x,fmaf(f.a.y,v,f.a.w)),fmaf(v,fmaf(f.a.z,v,f.b.x),f.b.y));}
  else next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);
  encodedBlend<Kind,Step>(power,color,done[0],T[0],C[0]);encodedBlend<Kind,Step>(next,color,done[1],T[1],C[1]);
}
