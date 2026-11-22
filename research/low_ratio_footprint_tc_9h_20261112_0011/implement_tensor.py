"""基于已读TC-GS公式自行实现WMMA候选，不引入官方训练或投影代码。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/DEV_CONTROLLER04.json').read_text())['state']=='complete'
    snapshot=HERE/'development_v3';snapshot.mkdir();shutil.copytree(HERE/'source',snapshot/'source')
    shutil.copy2(HERE/'build/mycartgs_footprint_ninehour_v1.so',snapshot/'mycartgs_footprint_ninehour_v1.so')
    shutil.copy2(HERE/'render_runtime.py',snapshot/'render_runtime.py')
    shutil.copy2(HERE/'tensor_pixels.cuh',HERE/'source/tensor_pixels.cuh')
    path=HERE/'source/graph_main.cu';s=path.read_text()
    s=s.replace('#include "approx_pixels.cuh"','#include "approx_pixels.cuh"\n#include "tensor_pixels.cuh"')
    s=s.replace('pixel_mode<=50','pixel_mode<=54')
    anchor=next(line for line in s.splitlines() if 'if(pixel_mode_==50)renderFoldedPixels<32>' in line)
    extra='\n'.join(f'      if(pixel_mode_=={18+m})renderTensorPixels<{m}><<<grid,128,0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);' for m in range(33,37))
    s=s.replace(anchor,anchor+'\n'+extra);path.write_text(s)
    path=HERE/'render_runtime.py';s=path.read_text();needle='\n\n\n@lru_cache'
    s=s.replace(needle,'\nMODES.update('+repr({33:'tensor_half',34:'tensor_half_stop',35:'tensor_tf32',36:'tensor_tf32_stop'})+')'+needle);path.write_text(s)
    print('TENSOR_RENDER_CANDIDATES_ADDED',4)


if __name__=='__main__':main()
