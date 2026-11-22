"""在独立副本研究8行/4行子Tile；已冻结源码与二进制保持不动。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    target=HERE/'late_candidates/microtile/source'
    if not target.exists():shutil.copytree(HERE/'source',target)
    path=target/'sparse_inference.cu';source=path.read_text(encoding='utf-8')
    assert 'renderMicrotile' not in source
    begin=source.index('template<int Pixels> __global__ void renderMultiPixel(')
    end=source.index('\n// Tile成员只在本Block',begin)
    kernel=source[begin:end]
    kernel=kernel.replace('template<int Pixels> __global__ void renderMultiPixel(',
        'template<int Slices> __global__ void renderMicrotile(')
    kernel=kernel.replace('constexpr int Rows=16/Pixels,Threads=256/Pixels;',
        'constexpr int Pixels=1,Rows=16/Slices,Threads=256/Slices;')
    kernel=kernel.replace('int tile=blockIdx.y*gridDim.x+blockIdx.x,thread=',
        'int tile=(blockIdx.y/Slices)*gridDim.x+blockIdx.x,thread=')
    kernel=kernel.replace('y=blockIdx.y*16+threadIdx.y;',
        'y=(blockIdx.y/Slices)*16+(blockIdx.y%Slices)*Rows+threadIdx.y;')
    kernel=kernel.replace('int64_t(gridDim.x)*gridDim.y*words+1',
        'int64_t(gridDim.x)*(gridDim.y/Slices)*words+1')
    source=source[:end]+'\n'+kernel+source[end:]
    source=source.replace('render_variant<=9','render_variant<=11')
    needle='          else if(render_variant==8)'
    assert source.count(needle)==1
    dispatch=''
    for variant,slices in [(10,2),(11,4)]:
        dispatch+=f'''          else if(render_variant=={variant})renderMicrotile<{slices}><<<dim3(grid.x,grid.y*{slices}),dim3(16,{16//slices}),0,stream>>>(W,H,bits,words,directory,groups,sorted_ids,
              lists,counts,capacity,xy,conic,feature,b.data_ptr<float>(),out.data_ptr<float>(),key_out);
'''
    source=source.replace(needle,dispatch+needle)
    source=source.replace('SixHourRasterizer','MicrotileStudyRasterizer')
    path.write_text(source,encoding='utf-8')
    print('MICROTILE_SNAPSHOT_CREATED',target)


if __name__=='__main__':main()
