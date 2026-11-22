// 每warp独立处理一个Tile，每线程8个不同像素；4个Tile共享一CTA的启动和shared容量。
// 二次型沿y递推，颜色/T保持每像素原深度顺序。新增舍入误差单独验收。
template<int Mode,int Pixels=8,bool Diagnose=false>
__global__ void renderWarpTilePixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const uint32_t* depth_keys,const GraphInputs* inputs,unsigned* flags=nullptr,
    unsigned long long* interactions=nullptr,float* diagnostic_image=nullptr){
  constexpr int Warps=4,Cache=64;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32,tiles_x=(W+15)/16,tiles_y=(H+15)/16;
  constexpr int TileWarps=8/Pixels,Rows=16/Pixels;
  const int tile=(blockIdx.x*Warps+warp)/TileWarps,sub=(blockIdx.x*Warps+warp)%TileWarps;
  if(tile>=tiles_x*tiles_y)return;
  const int tile_x=tile%tiles_x,tile_y=tile/tiles_x;
  const int x=tile_x*16+lane%16,y=tile_y*16+sub*2+lane/16;
  const float lx=lane%16-7.5f,ly=sub*2+lane/16-7.5f,cx=tile_x*16+7.5f,cy=tile_y*16+7.5f;
  const float* bg=inputs->background;float* out=Diagnose?diagnostic_image:inputs->image;
  float T[Pixels],C[Pixels][3];bool inside[Pixels],done[Pixels];unsigned long long visits=0;
  #pragma unroll
  for(int p=0;p<Pixels;++p){inside[p]=x<W&&y+p*Rows<H;done[p]=!inside[p];T[p]=1.f;C[p][0]=C[p][1]=C[p][2]=0.f;}
  if(depth_keys[0]==0xffffffff||bits[int64_t(tiles_x)*tiles_y*words+1]){
    #pragma unroll
    for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=bg[ch];
    return;
  }
  const int count=counts[tile];
  if(count<=capacity){
    __shared__ float4 abc[Warps][Cache];__shared__ float2 ef[Warps][Cache];
    __shared__ float3 rgb[Warps][Cache];__shared__ unsigned unsafe[Warps];
    for(int base=0;base<count;base+=Cache){
      const int batch=min(Cache,count-base);
      if(!lane)unsafe[warp]=0;
      __syncwarp();
      for(int j=lane;j<batch;j+=32){
        const int id=lists[tile*capacity+base+j];const auto f=foldGaussian(xy[id],conic[id],cx,cy);
        bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
        const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
        for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
        if(!safe)atomicOr(unsafe+warp,1u);
        abc[warp][j]=f.a;ef[warp][j]=f.b;rgb[warp][j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
      }
      __syncwarp();
      for(int j=0;j<batch;++j){
        if((j&7)==0){
          bool finished=true;
          #pragma unroll
          for(int p=0;p<Pixels;++p)finished=finished&&done[p];
          if(__all_sync(0xffffffff,finished))break;
        }
        const int id=lists[tile*capacity+base+j];
        if(unsafe[warp]){
          #pragma unroll
          for(int p=0;p<Pixels;++p){const float before=T[p];
            compositeScaling<9>(xy[id],conic[id],rgb[warp][j],{float(x),float(y+p*Rows)},done[p],T[p],C[p],-CUDART_INF_F);
            if constexpr(Diagnose)if(T[p]!=before){atomicOr(flags+id,1u);++visits;}}
        }else{
          const float4 a=abc[warp][j];const float2 b=ef[warp][j];const float3 color=rgb[warp][j];
          float power=fmaf(a.x,lx*lx,fmaf(a.z,ly*ly,fmaf(a.y,lx*ly,fmaf(a.w,lx,fmaf(b.x,ly,b.y)))));
          float increment=Rows*(a.y*lx+a.z*(2.f*ly+Rows)+b.x);
          const float second=2.f*Rows*Rows*a.z;
          #pragma unroll
          for(int p=0;p<Pixels;++p){
            if constexpr(Diagnose)debugBlendFolded<Mode>(power,color,done[p],T[p],C[p],id,flags,&visits);
            else blendFolded<Mode>(power,color,done[p],T[p],C[p]);
            power+=increment;increment+=second;
          }
        }
      }
      __syncwarp();
      bool finished=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)finished=finished&&done[p];
      if(__all_sync(0xffffffff,finished))break;
    }
  }else{
    for(int group=0;group<groups;++group){
      bool finished=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)finished=finished&&done[p];
      if(__all_sync(0xffffffff,finished))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty){const int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members){const int rank=w*32+__ffs(members)-1;members&=members-1;const int id=sorted[rank];
          #pragma unroll
          for(int p=0;p<Pixels;++p){const float before=T[p];
            compositeScaling<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},
                {float(x),float(y+p*Rows)},done[p],T[p],C[p],-CUDART_INF_F);
            if constexpr(Diagnose)if(T[p]!=before){atomicOr(flags+id,1u);++visits;}}
        }
      }
    }
  }
  if constexpr(Diagnose)if(visits)atomicAdd(interactions,visits);
  #pragma unroll
  for(int p=0;p<Pixels;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+p*Rows)*W+x]=C[p][ch]+T[p]*bg[ch];
}
