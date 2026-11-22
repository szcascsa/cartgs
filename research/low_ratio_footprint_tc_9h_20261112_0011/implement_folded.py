"""首批完成后保留完整回退版本，再实现有出处的局部多项式/指数候选。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/DEV_CONTROLLER02.json').read_text())['state']=='complete'
    assert all(json.loads(p.read_text())['state'] in ('complete','failed') for p in (HERE/'reports/jobs').glob('*.json'))
    snapshot=HERE/'development_v1';snapshot.mkdir();shutil.copytree(HERE/'source',snapshot/'source')
    shutil.copy2(HERE/'build/mycartgs_footprint_ninehour_v1.so',snapshot/'mycartgs_footprint_ninehour_v1.so')
    shutil.copy2(HERE/'render_runtime.py',snapshot/'render_runtime.py')
    shutil.copy2(HERE/'folded_pixels.cuh',HERE/'source/folded_pixels.cuh')
    path=HERE/'source/approx_pixels.cuh';s=path.read_text()
    s=s.replace('co.w*(Mode==14?exp(power):__expf(power))','co.w*(Mode==14?exp(power):Mode==26?boundedExp<1>(fmaxf(-80.f,power)):Mode==27?boundedExp<2>(fmaxf(-80.f,power)):__expf(power))')
    s=s.replace('Mode==5||Mode==14?.01f:','Mode==5||Mode==14||Mode==24?.01f:Mode==25?.02f:')
    s=s.replace('if((j&7)==0)','if((j&(Mode==23||Mode==24?31:7))==0)')
    path.write_text(s)
    path=HERE/'source/graph_main.cu';s=path.read_text()
    s=s.replace('#include "approx_pixels.cuh"','#include "folded_pixels.cuh"\n#include "approx_pixels.cuh"')
    s=s.replace('pixel_mode<=32','pixel_mode<=45')
    anchor=next(line for line in s.splitlines() if 'if(pixel_mode_==32)renderApproxPixels<14>' in line)
    extra=[]
    for m in range(15,28):
        kind='renderFoldedPixels' if m<=22 else 'renderApproxPixels'
        extra.append(f'      if(pixel_mode_=={18+m}){kind}<{m}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
    s=s.replace(anchor,anchor+'\n'+'\n'.join(extra));path.write_text(s)
    path=HERE/'render_runtime.py';s=path.read_text();needle='\n\n\n@lru_cache'
    modes={15:'folded_fast',16:'folded_stop_1e2',17:'folded_stop_2e2',18:'folded_stop_3e3',19:'folded_bitexp',
           20:'folded_bitexp_stop',21:'folded_polyexp',22:'folded_polyexp_stop',23:'fast_exp_check32',24:'stop_1e2_check32',
           25:'stop_2e2',26:'direct_bitexp',27:'direct_polyexp'}
    assert needle in s;s=s.replace(needle,'\nMODES.update('+repr(modes)+')'+needle);path.write_text(s)
    print('FOLDED_CANDIDATES_ADDED',len(modes))


if __name__=='__main__':main()
