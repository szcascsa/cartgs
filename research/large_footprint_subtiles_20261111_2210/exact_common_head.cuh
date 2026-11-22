// F2 sass原序：ax=co.x*dx，yy=(co.z*dy)*dy，cross=(co.y*dx)*dy，
// sum=fma(dx,ax,yy)，power=fma(sum,-.5,-cross)。不先舍入ax*dx。
__device__ __forceinline__ void compositeExactPair(float2 point,float4 co,float3 color,
    float2* pix,bool* done,float* T,float C[2][3]) {
  const float dx=__fadd_rn(point.x,-pix[0].x);
  const float ax=__fmul_rn(co.x,dx),cross_x=__fmul_rn(co.y,dx);
  #pragma unroll
  for(int p=0;p<2;++p) {
    if(done[p])continue;
    const float dy=__fadd_rn(point.y,-pix[p].y);
    const float yy=__fmul_rn(dy,__fmul_rn(co.z,dy));
    const float cross=__fmul_rn(dy,cross_x);
    const float power=__fmaf_rn(__fmaf_rn(dx,ax,yy),-.5f,-cross);
    if(power>0.f)continue;
    const float alpha=min(.99f,co.w*exp(power));
    if(alpha<1.f/255.f)continue;
    const float next=T[p]*(1-alpha);
    if(next<.0001f){done[p]=true;continue;}
    C[p][0]+=color.x*alpha*T[p];C[p][1]+=color.y*alpha*T[p];C[p][2]+=color.z*alpha*T[p];
    T[p]=next;
  }
}
