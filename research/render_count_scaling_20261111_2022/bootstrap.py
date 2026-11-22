"""独立复制渲染研究源码，保留所有旧版本与生产文件。"""
from pathlib import Path
import hashlib
import json
import re
import shutil

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'large_footprint_20261110_0912'
target = HERE / 'source'
assert not target.exists()
shutil.copytree(OLD / 'source', target)
for name in ('launch.py', 'job_runner.py'):
    shutil.copy2(HERE.parent / 'tensor_core_inference_20261110_2200/preparation_phase' / name, HERE / name)
for name in ('build', 'reports', 'logs'):
    (HERE / name).mkdir(exist_ok=True)
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (OLD / 'source').rglob('*') if p.is_file()}
manifest[str(OLD / 'build_v4/large_footprint_graph_v4.so')] = hashlib.sha256(
    (OLD / 'build_v4/large_footprint_graph_v4.so').read_bytes()).hexdigest()
(HERE / 'reports/BASELINE_SHA256.json').write_text(json.dumps(manifest, indent=2))

main = target / 'graph_main.cu'
text = main.read_text()
text = text.replace('#include "preprocess_tasks.cuh"', '#include "preprocess_tasks.cuh"\n#include "scaling_pixels.cuh"')
text = text.replace('FootprintGraphRasterizerV4', 'CountScalingRasterizer')
text = text.replace('pixel_mode<=5', 'pixel_mode<=9')
old = 'if(pixel_mode_==4)renderDirectPixels<2>'
new = """if(pixel_mode_>=6) {
      if(pixel_mode_==6)renderScalingPixels<1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==7)renderScalingPixels<2><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==8)renderScalingPixels<3><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
      if(pixel_mode_==9)renderScalingPixels<4><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);
    } else if(pixel_mode_==4)renderDirectPixels<2>"""
assert old in text
main.write_text(text.replace(old, new))

kernels = (target / 'graph_kernels.cuh').read_text()
begin = kernels.index('template<int Pixels> __global__ void renderMultiPixel')
end = kernels.index('// Tile成员只在本Block枚举一次', begin)
pixel = kernels[begin:end]
pixel = pixel.replace('template<int Pixels> __global__ void renderMultiPixel',
                      'template<int Mode> __global__ void renderScalingPixels')
pixel = pixel.replace('constexpr int Rows=16/Pixels,Threads=256/Pixels;',
                      'constexpr int Pixels=2,Rows=8,Threads=128;')
pixel = pixel.replace('__shared__ float2 cached_xy[256];',
                      '__shared__ float cached_bound[Mode==2?256:1];__shared__ float2 cached_xy[256];')
pixel = pixel.replace('cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};',
                      'cached_color[j]={colors[id*3],colors[id*3+1],colors[id*3+2]};if(Mode==2)cached_bound[j]=alphaPowerBound(conic[id].w);')
pixel = pixel.replace('float3 color=cached_color[j];',
                      'float3 color=cached_color[j];float bound=Mode==2?cached_bound[j]:-CUDART_INF_F;')
pixel = pixel.replace('float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};',
                      'float3 color={colors[id*3],colors[id*3+1],colors[id*3+2]};float bound=-CUDART_INF_F;')
pixel, calls = re.subn(r'#pragma unroll\n\s*for\(int p=0;p<Pixels;\+\+p\)compositeValues\(point,co,color,pix\[p\],done\[p\],T\[p\],C\[p\]\);',
                      'compositeScalingPair<Mode>(point,co,color,pix,done,T,C,bound);',pixel)
assert calls==2
assert 'compositeValues(point,co,color,pix[p]' not in pixel
header = (HERE / 'scaling_pixels_head.cuh').read_text()
(target / 'scaling_pixels.cuh').write_text(header + '\n' + pixel)
print('INDEPENDENT_SOURCE_READY', len(manifest), flush=True)
