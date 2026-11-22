// bit-exp在原power早拒绝阈值处仍低于Alpha阈值；负端限幅后可合并两次拒绝。
__device__ __forceinline__ float mergedFoldedAlpha(float power,bool done){
  if(done)return 0.f;
  int bits=__float2int_rz(fmaf(fmaxf(power,-6.f),12102203.161561486f,1064970947.f));
  bits=min(bits,__float_as_int(.99f));
  return bits<__float_as_int(1.f/255.f)?0.f:__int_as_float(bits);
}

template<int Kind> __device__ __forceinline__ void branchlessBlend(
    float power,float3 color,bool& done,float& T,float* C){
  float alpha;
  if constexpr(Kind==4){const float value=mergedFoldedAlpha(power,false);alpha=done?0.f:value;}
  else alpha=mergedFoldedAlpha(power,done);
  if constexpr(Kind==1){
    if(alpha==0.f)return;
    const float next=T*(1.f-alpha);if(next<.0001f){done=true;return;}
    const float weight=alpha*T;C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
    T=next;if(T<.01f)done=true;
  }else{
    float next=T*(1.f-alpha),weight=alpha*T;
    if constexpr(Kind==2){const bool crossing=next<.0001f;done=done||crossing;weight=crossing?0.f:weight;next=crossing?T:next;}
    C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
    T=next;if(T<.01f)done=true;
  }
}

template<int Kind> __device__ __forceinline__ void branchlessPair(FoldedGaussian f,float3 color,
    float x,float y,bool* done,float* T,float C[2][3]){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+8.f*(f.a.y*x+f.a.z*(2.f*y+8.f)+f.b.x);
  branchlessBlend<Kind>(power,color,done[0],T[0],C[0]);branchlessBlend<Kind>(next,color,done[1],T[1],C[1]);
}
