// 一个线程处理一个Tile的有序中心尾部，跨Tile并行；每帧重算，不缓存旧视角。
__global__ void prepareSerialTail(int W,int H,const uint32_t* lists,const uint32_t* counts,int capacity,
    const float2* xy,const float4* conic,const float* colors,const GraphInputs* inputs,float* tail){
  const int tile=blockIdx.x*blockDim.x+threadIdx.x,tiles_x=(W+15)/16,tiles_y=(H+15)/16;
  if(tile>=tiles_x*tiles_y||counts[tile]>unsigned(capacity))return;
  const int count=counts[tile];const float cx=(tile%tiles_x)*16+7.5f,cy=(tile/tiles_x)*16+7.5f;
  float3 value={inputs->background[0],inputs->background[1],inputs->background[2]};
  const int64_t base=int64_t(tile)*(capacity+1)*3;
  tail[base+count*3]=value.x;tail[base+count*3+1]=value.y;tail[base+count*3+2]=value.z;
  for(int j=count-1;j>=0;--j){
    const int id=lists[tile*capacity+j];const auto f=foldGaussian(xy[id],conic[id],cx,cy);
    float alpha=0.f;
    if(f.b.y>=-5.541263545158426f){alpha=fminf(.99f,boundedExp<1>(f.b.y));if(alpha<1.f/255.f)alpha=0.f;}
    if(alpha>0.f){value={fmaf(1.f-alpha,value.x,colors[id*3]*alpha),
        fmaf(1.f-alpha,value.y,colors[id*3+1]*alpha),fmaf(1.f-alpha,value.z,colors[id*3+2]*alpha)};}
    tail[base+j*3]=value.x;tail[base+j*3+1]=value.y;tail[base+j*3+2]=value.z;
  }
}

template<int StopPercent,bool Diagnose>
__device__ __forceinline__ void blendSerialTail(float power,float3 color,bool& done,float& T,float* C,
    const float* tail,unsigned id,unsigned* flags,unsigned long long* visits,
    unsigned long long* pixels,unsigned long long* skipped,int remaining){
  if(done||power<-5.541263545158426f)return;
  const float alpha=fminf(.99f,boundedExp<1>(power));if(alpha<1.f/255.f)return;
  const float next=T*(1-alpha);if(next<.0001f){done=true;return;}
  const float weight=alpha*T;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);T=next;
  if constexpr(Diagnose){atomicOr(flags+id,1u);++*visits;}
  if(T<StopPercent*.01f){
    C[0]=fmaf(T,tail[0],C[0]);C[1]=fmaf(T,tail[1],C[1]);C[2]=fmaf(T,tail[2],C[2]);T=0.f;done=true;
    if constexpr(Diagnose){++*pixels;*skipped+=remaining;}
  }
}
