// 只筛除保守证明Alpha无贡献的子块；原Pixel表达式和累加顺序不变。
__device__ __forceinline__ float intervalProductLower(float a,float b,float c,float d) {
  return fminf(fminf(__fmul_rd(a,c),__fmul_rd(a,d)),
               fminf(__fmul_rd(b,c),__fmul_rd(b,d)));
}
__device__ __forceinline__ bool mayContributeRect(float2 mean,float4 co,
    int xmin,int ymin,int xmax,int ymax) {
  if(!(co.w>=0.f&&co.w<=1.f&&co.x>0.f&&co.z>0.f))return true;
  const float lx=__fsub_rd(mean.x,float(xmax)),ux=__fsub_ru(mean.x,float(xmin));
  const float ly=__fsub_rd(mean.y,float(ymax)),uy=__fsub_ru(mean.y,float(ymin));
  const float ax=lx>0.f?lx:ux<0.f?-ux:0.f;
  const float ay=ly>0.f?ly:uy<0.f?-uy:0.f;
  const float xx=__fmul_rd(__fmul_rd(co.x,ax),ax);
  const float yy=__fmul_rd(__fmul_rd(co.z,ay),ay);
  const float lo=co.y>=0.f?__fmul_rd(co.y,lx):__fmul_rd(co.y,ux);
  const float hi=co.y>=0.f?__fmul_ru(co.y,ux):__fmul_ru(co.y,lx);
  const float cross=intervalProductLower(lo,hi,ly,uy);
  const float upper=__fsub_ru(__fmul_ru(-.5f,__fadd_rd(xx,yy)),cross);
  // 外扩误差覆盖原编译器FMA分组、乘法舍入和FTZ；不确定或overflow一律保留。
  const float magnitude=fabsf(xx)+fabsf(yy)+fabsf(cross)+1.f;
  const float margin=__fmul_ru(magnitude,0x1p-16f);
  const float bound=__fadd_ru(upper,margin);
  const float cutoff=co.w<=.00390625f?0.f:co.w<=.0078125f?-1.f:
      co.w<=.015625f?-1.5f:co.w<=.0625f?-3.f:co.w<=.125f?-4.f:co.w<=.5f?-5.f:-6.f;
  return !(isfinite(bound)&&bound<cutoff);
}

// Kind 0: 8x8子块每Warp稳定压缩；Kind 1: 4x4子块稳定压缩；
// Kind 2: 不压缩，仅共享8x8掩码；Kind 3: 一次性精确扫描区域以作收益上界诊断。
template<int Kind> __global__ void renderSubtilePixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const uint32_t* keys,const GraphInputs* inputs) {
  constexpr int Side=Kind==1?4:8;
  constexpr int Regions=256/(Side*Side),PerRegion=Side*Side;
  const int thread=threadIdx.x,lane=thread&31,warp=thread>>5;
  const int tile=blockIdx.y*gridDim.x+blockIdx.x,ox=blockIdx.x*16,oy=blockIdx.y*16;
  const float* bg=inputs->background;float* out=inputs->image;
  const int pixel0=thread*2,pixel1=pixel0+1;
  // 像素排列为连续子块，两条独立原始合成链。
  int linear[2]={pixel0,pixel1};float2 pix[2];bool inside[2],done[2];float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  #pragma unroll
  for(int p=0;p<2;++p) {
    int r=linear[p]/PerRegion,j=linear[p]%PerRegion;
    int x=ox+(r%(16/Side))*Side+j%Side,y=oy+(r/(16/Side))*Side+j/Side;
    pix[p]={float(x),float(y)};inside[p]=x<W&&y<H;done[p]=!inside[p];
  }
  if(keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]) {
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+int(pix[p].y)*W+int(pix[p].x)]=bg[ch];
    return;
  }
  const unsigned count=counts[tile];
  if(count<=unsigned(capacity)) {
    __shared__ float2 cached_xy[256];
    __shared__ float4 cached_co[256];
    __shared__ float3 cached_rgb[256];
    __shared__ unsigned masks[256];
    __shared__ uint16_t members[Regions][256];
    __shared__ unsigned sizes[Regions];
    for(int j=thread;j<int(count);j+=128) {
      const int id=lists[tile*capacity+j];cached_xy[j]=xy[id];cached_co[j]=conic[id];
      cached_rgb[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};
    }
    __syncthreads();
    if(Kind==2) {
      for(int j=thread;j<int(count);j+=128) {
        unsigned mask=0;
        #pragma unroll
        for(int r=0;r<4;++r) {
          const int xmin=ox+(r&1)*8,ymin=oy+(r>>1)*8;
          if(mayContributeRect(cached_xy[j],cached_co[j],xmin,ymin,xmin+7,ymin+7))mask|=1u<<r;
        }
        masks[j]=mask;
      }
    } else {
      for(int r=warp;r<Regions;r+=4) {
        const int xmin=ox+(r%(16/Side))*Side,ymin=oy+(r/(16/Side))*Side;
        unsigned carry=0;
        for(int base=0;base<int(count);base+=32) {
          const int j=base+lane;
          bool valid=j<int(count)&&mayContributeRect(cached_xy[j],cached_co[j],xmin,ymin,xmin+Side-1,ymin+Side-1);
          if(Kind==3&&valid) {
            // 只供诊断：精确是否存在有效像素，不作为预设速度候选。
            valid=false;
            for(int p=0;p<PerRegion;++p) {
              float2 point={float(xmin+p%Side),float(ymin+p/Side)};
              if(alphaValue(cached_xy[j],cached_co[j],point)>=0.f){valid=true;break;}
            }
          }
          const unsigned active=__ballot_sync(0xffffffff,valid);
          if(valid)members[r][carry+__popc(active&((1u<<lane)-1))]=j;
          carry+=__popc(active);
        }
        if(!lane)sizes[r]=carry;
      }
    }
    __syncthreads();
    const int region=linear[0]/PerRegion;
    const unsigned region_lanes=__match_any_sync(0xffffffff,region);
    const int iterations=Kind==2?int(count):int(sizes[region]);
    #pragma unroll 8
    for(int i=0;i<iterations;++i) {
      if((i&15)==0&&__all_sync(region_lanes,done[0]&&done[1]))break;
      const int j=Kind==2?i:members[region][i];
      if(Kind!=2||(masks[j]&(1u<<region))) {
        const float2 point=cached_xy[j];const float4 co=cached_co[j];const float3 color=cached_rgb[j];
        #pragma unroll
        for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
      }
    }
  } else {
    // 超过短列表容量时完整沿稳定目录枚举；不截断。
    for(int group=0;group<groups;++group) {
      if(__all_sync(0xffffffff,done[0]&&done[1]))break;
      unsigned directory_word=directory[tile*groups+group];
      while(directory_word) {
        int word=group*32+__ffs(directory_word)-1;directory_word&=directory_word-1;
        unsigned mask=bits[tile*words+word];
        while(mask) {
          int rank=word*32+__ffs(mask)-1;mask&=mask-1;int id=sorted[rank];
          float2 point=xy[id];float4 co=conic[id];float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};
          #pragma unroll
          for(int p=0;p<2;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);
        }
      }
    }
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)
    out[ch*H*W+int(pix[p].y)*W+int(pix[p].x)]=C[p][ch]+T[p]*bg[ch];
}

// 仅供计时外诊断，不影响render的任何执行节点。
__global__ void subtileDiagnostics(int tiles,int side,int W,int H,const uint32_t* lists,const uint32_t* counts,
    const float2* xy,const float4* conic,unsigned long long* counters,bool verify) {
  int index=blockIdx.x*blockDim.x+threadIdx.x,tile=index/256,j=index%256;
  if(tile>=tiles)return;
  const unsigned count=counts[tile];
  if(!j){atomicAdd(counters,static_cast<unsigned long long>(count));atomicAdd(counters+6,1ull);
    if(count>256)atomicAdd(counters+7,1ull);}
  if(count>256||unsigned(j)>=count)return;
  const int id=lists[tile*256+j],tx=tile%((W+15)/16),ty=tile/((W+15)/16);
  unsigned tests=0,keep=0,reject=0,wrong=0;
  for(int y=0;y<16;y+=side)for(int x=0;x<16;x+=side) {
    int xmin=tx*16+x,ymin=ty*16+y;
    ++tests;
    if(mayContributeRect(xy[id],conic[id],xmin,ymin,xmin+side-1,ymin+side-1)){++keep;continue;}
    ++reject;
    if(verify)for(int py=ymin;py<min(H,ymin+side);++py)for(int px=xmin;px<min(W,xmin+side);++px)
      if(alphaValue(xy[id],conic[id],make_float2(px,py))>=0.f)++wrong;
  }
  atomicAdd(counters+1,static_cast<unsigned long long>(tests));
  atomicAdd(counters+2,static_cast<unsigned long long>(keep));
  atomicAdd(counters+3,static_cast<unsigned long long>(reject));
  atomicAdd(counters+4,static_cast<unsigned long long>(wrong));
  atomicAdd(counters+5,static_cast<unsigned long long>(keep*side*side));
}
