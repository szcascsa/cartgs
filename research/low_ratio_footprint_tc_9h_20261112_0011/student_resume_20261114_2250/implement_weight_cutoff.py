"""独立近似：保留原Alpha/T递推，裁掉最终权重alpha*T低于门限的颜色FMA。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_weight_cutoff'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_weight_cutoff').mkdir()
    for name in ['encoded_sign_pixels.cuh', 'encoded_sign_diagnosed.cuh']:
        text = (root / name).read_text().replace('encodedSign', 'weightCut').replace('EncodedSign', 'WeightCut')
        text = text.replace('template<int Kind,int Step>', 'template<int Kind,int Step,int Cut>')
        text = text.replace('template<int Mode,int Kind,int Layout>', 'template<int Mode,int Kind,int Layout,int Cut>')
        text = text.replace('<Kind,Step>', '<Kind,Step,Cut>')
        anchor = '  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);'
        assert text.count(anchor) == 1
        text = text.replace(anchor, '''  constexpr float threshold=Cut==1?1e-4f:Cut==2?3e-4f:Cut==3?1e-3f:3e-3f;
  if(weight>=threshold){C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);}''')
        if name.endswith('_diagnosed.cuh'):
            text = text.replace('if(weight>0.f)', 'if(weight>=threshold)')
        (root / name.replace('encoded_sign', 'weight_cutoff')).write_text(text)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "weight_cutoff_pixels.cuh"\n#include "weight_cutoff_diagnosed.cuh"')
    text = text.replace('pixel_mode<=202', 'pixel_mode<=342').replace('pixel_mode_<=202', 'pixel_mode_<=342')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    calls, diagnostic = [], []
    for mode, cut in [(321, 1), (322, 2), (323, 3), (324, 4)]:
        calls.append(f'      if(pixel_mode_=={mode+18})renderWeightCutPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostic.append(f'    if(pixel_mode_=={mode+18})renderWeightCutDiagnosedPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text = text.replace(anchor, anchor + '\n' + '\n'.join(calls))
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '\n'.join(diagnostic) + '\n' + anchor)
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentWeightCutoffRasterizer'))
    print('WEIGHT_CUTOFF_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
