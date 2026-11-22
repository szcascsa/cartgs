"""独立近似候选：提高残余透射率早停阈值，保留原182及F2。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_transmittance'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_transmittance').mkdir()
    for name in ['encoded_sign_pixels.cuh', 'encoded_sign_diagnosed.cuh']:
        text = (root / name).read_text().replace('encodedSign', 'stop').replace('EncodedSign', 'Stop')
        text = text.replace('template<int Kind,int Step>', 'template<int Kind,int Step,int Stop>')
        text = text.replace('template<int Mode,int Kind,int Layout>', 'template<int Mode,int Kind,int Layout,int Stop>')
        text = text.replace('<Kind,Step>', '<Kind,Step,Stop>')
        assert text.count('next<.01f') == 1
        text = text.replace('next<.01f', 'next<(Stop==1?.015f:Stop==2?.02f:Stop==3?.03f:.05f)')
        (root / name.replace('encoded_sign', 'stop')).write_text(text)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "stop_pixels.cuh"\n#include "stop_diagnosed.cuh"')
    text = text.replace('pixel_mode<=202', 'pixel_mode<=242').replace('pixel_mode_<=202', 'pixel_mode_<=242')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    calls, diagnostic = [], []
    for mode, stop in [(221, 1), (222, 2), (223, 3), (224, 4)]:
        calls.append(f'      if(pixel_mode_=={mode+18})renderStopPixels<38,3,1,{stop}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostic.append(f'    if(pixel_mode_=={mode+18})renderStopDiagnosedPixels<38,3,1,{stop}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text = text.replace(anchor, anchor + '\n' + '\n'.join(calls))
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '\n'.join(diagnostic) + '\n' + anchor)
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentTransmittanceRasterizer'))
    print('TRANSMITTANCE_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
