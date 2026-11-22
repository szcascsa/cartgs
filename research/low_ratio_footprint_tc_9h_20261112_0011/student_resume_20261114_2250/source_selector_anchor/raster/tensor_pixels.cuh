#include <mma.h>
#include <cuda_fp16.h>

// 独立探索：16成员×16像素幂指数矩阵；仅改变指数求值，颜色/T仍按成员原序。
template<int Mode> __device__ __forceinline__ void blendTensorPower(float power,float3 color,bool& done,float& T,float* C){
  if(done||power<-5.541263545158426f)return;
  const float alpha=min(.99f,__expf(power));if(alpha<1.f/255.f)return;
  const float next=T*(1-alpha);if(next<.0001f){done=true;return;}
  const float weight=alpha*T;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);T=next;
  if((Mode==34||Mode==36)&&T<.01f)done=true;
}

template<int Mode> __global__ void renderTensorPixels(int W,int H,const uint32_t* bits,int words,
    const uint32_t* directory,int groups,const uint32_t* sorted,const uint32_t* lists,
    const uint32_t* counts,int capacity,const float2* xy,const float4* conic,const float* colors,
    const float*,float*,const uint32_t* depth_keys,const GraphInputs* inputs){
  constexpr bool TF=Mode>=35;constexpr int D=TF?8:16;
  using Storage=typename std::conditional<TF,float,half>::type;
  namespace wmma=nvcuda::wmma;
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,tile=blockIdx.y*gridDim.x+blockIdx.x;
  int pi[2]={warp*64+lane,warp*64+lane+32},xx[2],yy[2];bool inside[2],done[2];
  float T[2]={1.f,1.f},C[2][3]={{0.f},{0.f}};
  #pragma unroll
  for(int p=0;p<2;++p){xx[p]=blockIdx.x*16+pi[p]%16;yy[p]=blockIdx.y*16+pi[p]/16;
    inside[p]=xx[p]<W&&yy[p]<H;done[p]=!inside[p];}
  const float* bg=inputs->background;float* out=inputs->image;
  if(depth_keys[0]==0xffffffff||bits[int64_t(gridDim.x)*gridDim.y*words+1]){
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+yy[p]*W+xx[p]]=bg[ch];return;
  }
  const int count=counts[tile];const float cx=blockIdx.x*16+7.5f,cy=blockIdx.y*16+7.5f;
  // 实际Tile成员不足以摊薄MMA或短列表溢出：保持同一候选的标量求值，而非按场景/Ratio切换。
  if(count<16||count>capacity){
    for(int group=0;group<groups;++group){
      if(__all_sync(0xffffffff,done[0]&&done[1]))break;
      unsigned nonempty=directory[tile*groups+group];
      while(nonempty){const int w=group*32+__ffs(nonempty)-1;nonempty&=nonempty-1;unsigned members=bits[tile*words+w];
        while(members){int rank=w*32+__ffs(members)-1;members&=members-1;int id=sorted[rank];
          #pragma unroll
          for(int p=0;p<2;++p){
            float2 d={xy[id].x-xx[p],xy[id].y-yy[p]};const float4 co=conic[id];
            float power=-.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y+logf(co.w);
            blendTensorPower<Mode>(power,{colors[id*3],colors[id*3+1],colors[id*3+2]},done[p],T[p],C[p]);
          }
        }
      }
    }
  }else{
    __shared__ __align__(32) Storage A[256*D];
    __shared__ __align__(32) Storage B[16*D];
    __shared__ __align__(32) float product[4096];
    __shared__ float2 points[16];__shared__ float4 cones[16];__shared__ float3 rgb[16];
    __shared__ unsigned safe[16];
    for(int i=tid;i<256*D;i+=128){
      const int row=i/D,k=i%D;const float x=row%16-7.5f,y=row/16-7.5f;
      const float v=k==0?x*x:k==1?x*y:k==2?y*y:k==3?x:k==4?y:k==5?1.f:0.f;
      if constexpr(TF)A[i]=wmma::__float_to_tf32(v);else A[i]=__float2half_rn(v);
    }
    __syncthreads();
    for(int base=0;base<count;base+=16){
      if(__syncthreads_count(done[0]&&done[1])==128)break;
      if(tid<16){
        const int j=base+tid;
        if(j<count){const int id=lists[tile*capacity+j];points[tid]=xy[id];cones[tid]=conic[id];rgb[tid]={colors[id*3],colors[id*3+1],colors[id*3+2]};
          const auto f=foldGaussian(xy[id],conic[id],cx,cy);const float v[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
          bool good=conic[id].w>0.f&&conic[id].w<=1.f;
          for(int k=0;k<6;++k)good=good&&isfinite(v[k])&&fabsf(v[k])<10000.f;
          safe[tid]=good;
          for(int k=0;k<D;++k){const float value=good&&k<6?v[k]:0.f;
            if constexpr(TF)B[tid*D+k]=wmma::__float_to_tf32(value);else B[tid*D+k]=__float2half_rn(value);}
        }else{safe[tid]=0;for(int k=0;k<D;++k){if constexpr(TF)B[tid*D+k]=0.f;else B[tid*D+k]=__float2half_rn(0.f);}}
      }
      __syncthreads();
      if(!__all_sync(0xffffffff,done[0]&&done[1])){
        if constexpr(TF){
          wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b;
          wmma::load_matrix_sync(b,B,D);
          #pragma unroll
          for(int chunk=0;chunk<4;++chunk){
            wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a;
            wmma::fragment<wmma::accumulator,16,16,8,float> result;wmma::fill_fragment(result,0.f);
            wmma::load_matrix_sync(a,A+(warp*64+chunk*16)*D,D);wmma::mma_sync(result,a,b,result);
            wmma::store_matrix_sync(product+warp*1024+chunk*256,result,16,wmma::mem_row_major);
          }
        }else{
          wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> b;wmma::load_matrix_sync(b,B,D);
          #pragma unroll
          for(int chunk=0;chunk<4;++chunk){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
            wmma::fragment<wmma::accumulator,16,16,16,float> result;wmma::fill_fragment(result,0.f);
            wmma::load_matrix_sync(a,A+(warp*64+chunk*16)*D,D);wmma::mma_sync(result,a,b,result);
            wmma::store_matrix_sync(product+warp*1024+chunk*256,result,16,wmma::mem_row_major);
          }
        }
      }
      __syncthreads();
      #pragma unroll
      for(int j=0;j<16;++j){
        if(base+j>=count)break;
        #pragma unroll
        for(int p=0;p<2;++p){
          if(done[p])continue;
          if(safe[j])blendTensorPower<Mode>(product[(warp*64+lane+p*32)*16+j],rgb[j],done[p],T[p],C[p]);
          else {
            const float4 co=cones[j];float2 pix={float(xx[p]),float(yy[p])};
            compositeScaling<9>(points[j],co,rgb[j],pix,done[p],T[p],C[p],-CUDART_INF_F);
          }
        }
      }
      __syncthreads();
    }
  }
  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+yy[p]*W+xx[p]]=C[p][ch]+T[p]*bg[ch];
}
