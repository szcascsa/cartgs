"""保持逐像素合成顺序，扩大独立Alpha指令并行；不减少有效Alpha工作。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_subtiles').exists()
shutil.copytree(source,HERE/'source_subtiles')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/subtiles_diagnostics.so')
text=(source/'scaling_pixels.cuh').read_text()
start=text.index('template<int Mode> __global__ void renderScalingPixels')
pixel=text[start:]
pixel=pixel.replace('template<int Mode> __global__ void renderScalingPixels',
                    'template<int Batch,int Pixels=2> __global__ void renderAlphaPipeline')
pixel=pixel.replace('constexpr int Pixels=2,Rows=8,Threads=128;',
                    'constexpr int Rows=16/Pixels,Threads=256/Pixels;')
pixel=pixel.replace('cached_bound[Mode==2?256:1];','unused_bound[1];')
pixel=pixel.replace('j+=128','j+=Threads')
pixel=pixel.replace('if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);','')
begin=pixel.index('    constexpr int Unroll=')
end=pixel.index('  } else {',begin)
loop="""    for(int base=0;base<count;base+=Batch) {
      bool all_done=true;
      #pragma unroll
      for(int p=0;p<Pixels;++p)all_done=all_done&&done[p];
      if(__all_sync(0xffffffff,all_done))break;
      float alphas[Batch][Pixels];float3 color[Batch];
      #pragma unroll
      for(int j=0;j<Batch;++j) {
        const bool valid=base+j<count;
        float2 point=valid?cached_xy[base+j]:make_float2(0,0);
        float4 co=valid?cached_conic[base+j]:make_float4(0,0,0,0);
        color[j]=valid?cached_color[base+j]:make_float3(0,0,0);
        #pragma unroll
        for(int p=0;p<Pixels;++p)
          alphas[j][p]=valid&&!done[p]?alphaValue(point,co,pix[p]):-1.f;
      }
      #pragma unroll
      for(int j=0;j<Batch;++j)if(base+j<count) {
        #pragma unroll
        for(int p=0;p<Pixels;++p)compositeAlpha(alphas[j][p],color[j],done[p],T[p],C[p]);
      }
    }
"""
pixel=pixel[:begin]+loop+pixel[end:]
pixel=pixel.replace('float bound=-CUDART_INF_F;','')
pixel=pixel.replace('compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);',
                    '#pragma unroll\n          for(int p=0;p<Pixels;++p)compositeValues(point,co,color,pix[p],done[p],T[p],C[p]);')
assert 'Mode' not in pixel
(source/'alpha_pipeline.cuh').write_text(pixel)
main=source/'graph_main.cu'
text=main.read_text().replace('#include "subtile_pixels.cuh"','#include "subtile_pixels.cuh"\n#include "alpha_pipeline.cuh"')
text=text.replace('pixel_mode<=22','pixel_mode<=28')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==22)renderSubtilePixels' in x)
extra='\n'.join(f'      if(pixel_mode_=={m})renderAlphaPipeline<{batch},{pixels}><<<grid,dim3(16,{16//pixels}),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
               for m,batch,pixels in [(23,2,2),(24,4,2),(25,8,2),(26,2,4),(27,4,4),(28,2,1)])
main.write_text(text.replace(line,line+'\n'+extra))
print('ALPHA_PIPELINE_READY',flush=True)
