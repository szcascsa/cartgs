"""独立可切换近似合成：仅exp/阈值，不改投影、身份、K、属性或排序。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    original=HERE/'source/scaling_pixels.cuh';text=original.read_text()
    head='''// 推理研究：FastExp与受控贡献/透射率阈值，Mode0仍由冻结F2作对照。
template<int Mode> __device__ __forceinline__ void compositeApprox(
    float2 point,float4 co,float3 color,float2 pix,bool& done,float& T,float* C,float) {
  if(done)return;
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
  if(power>0.f)return;
  const float alpha_cut=Mode==8?2.f/255.f:Mode==9?4.f/255.f:Mode==10?8.f/255.f:1.f/255.f;
  float alpha=min(.99f,co.w*(Mode==14?exp(power):__expf(power)));
  if(alpha<alpha_cut)return;
  const float next=T*(1-alpha);
  // 保留原F2在1e-4处的先验终止语义。额外近似提前退出在本成员正常累加之后。
  if(next<.0001f){done=true;return;}
  const float contribution=T*alpha;
  const float weight_cut=Mode==11?.0001f:Mode==12?.0005f:Mode==13?.001f:0.f;
  if(contribution>=weight_cut){
    C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  }
  T=next;
  const float stop=Mode==2?.0003f:Mode==3?.001f:Mode==4?.003f:Mode==5||Mode==14?.01f:Mode==6?.03f:Mode==7?.1f:0.f;
  if(stop>0.f&&T<stop)done=true;
}

template<int Mode> __device__ __forceinline__ void compositeApproxPair(
    float2 point,float4 co,float3 color,float2* pix,bool* done,float* T,float C[2][3],float bound) {
  #pragma unroll
  for(int p=0;p<2;++p)compositeApprox<Mode>(point,co,color,pix[p],done[p],T[p],C[p],bound);
}

'''
    kernel=text[text.index('template<int Mode> __global__ void renderScalingPixels'):]
    kernel=kernel.replace('renderScalingPixels','renderApproxPixels').replace('compositeScalingPair','compositeApproxPair')
    kernel=kernel.replace('Mode==2?256:1','1').replace('if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);','')
    kernel=kernel.replace('Mode==2?cached_bound[j]:-CUDART_INF_F','-CUDART_INF_F')
    kernel=kernel.replace('constexpr int Unroll=Mode==5?2:Mode==6?4:Mode==7?8:Mode==8?16:Mode==9?32:1;','constexpr int Unroll=32;')
    # 近似早停更频繁检查warp完成，独立不改变被累加像素的次序。
    kernel=kernel.replace('(j&31)==0','(j&7)==0')
    (HERE/'source/approx_pixels.cuh').write_text(head+kernel)
    main=HERE/'source/graph_main.cu';s=main.read_text()
    s=s.replace('BackgroundScalingRasterizer','NineHourRasterizer')
    s=s.replace('#include "packed_pixels.cuh"','#include "packed_pixels.cuh"\n#include "approx_pixels.cuh"')
    s=s.replace('pixel_mode<=18','pixel_mode<=32').replace('pixel_mode_==18?inputs:nullptr','pixel_mode_>=18?inputs:nullptr')
    anchor=next(line for line in s.splitlines() if 'if(pixel_mode_==18)renderScalingPixels<9>' in line)
    extra='\n'.join(f'      if(pixel_mode_=={18+m})renderApproxPixels<{m}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);' for m in range(1,15))
    assert anchor in s;s=s.replace(anchor,anchor+'\n'+extra);main.write_text(s)
    print('APPROX_CANDIDATES_IMPLEMENTED',14)


if __name__=='__main__':main()
