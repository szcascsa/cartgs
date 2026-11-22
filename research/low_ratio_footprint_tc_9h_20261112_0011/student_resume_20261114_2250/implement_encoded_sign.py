"""独立组合：bit-exp线性投影提升、T符号状态与warp映射。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
MODES={181:(3,0),182:(3,1),183:(2,1),184:(1,1)}


def main():
    root=HERE/'source_encoded_sign';shutil.copytree(HERE/'source_encoded',root);(HERE/'build_encoded_sign').mkdir()
    scale='''if constexpr(true){
        f.a.x*=12102203.161561486f;f.a.y*=12102203.161561486f;f.a.z*=12102203.161561486f;f.a.w*=12102203.161561486f;
        f.b.x*=12102203.161561486f;f.b.y=fmaf(f.b.y,12102203.161561486f,1064970947.f);
      }'''
    for name in ['sign_pixels.cuh','sign_diagnosed.cuh']:
        text=(HERE/'source_sign'/name).read_text()
        text=text.replace('signBlend','encodedSignBlend').replace('signPair','encodedSignPair').replace('signFallback','encodedSignFallback')
        text=text.replace('diagnosedSign','diagnosedEncodedSign').replace('renderSign','renderEncodedSign')
        text=text.replace('mergedFoldedAlpha(power,T<0.f)','encodedAlpha(power,T<0.f)')
        assert text.count('abc[j]=f.a')==1
        text=text.replace('abc[j]=f.a',scale+'\n      abc[j]=f.a')
        (root/name.replace('sign','encoded_sign')).write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "encoded_diagnosed.cuh"',
        '#include "encoded_diagnosed.cuh"\n#include "encoded_sign_pixels.cuh"\n#include "encoded_sign_diagnosed.cuh"')
    text=text.replace('pixel_mode<=192','pixel_mode<=202').replace('pixel_mode_<=192','pixel_mode_<=202')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==192)renderEncodedPixels' in line)
    launches=[];diagnostics=[]
    for mode,(kind,layout) in MODES.items():
        launches.append(f'      if(pixel_mode_=={mode+18})renderEncodedSignPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderEncodedSignDiagnosedPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(launches))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostics)+'\n'+anchor).replace('StudentEncodedRasterizer','StudentEncodedSignRasterizer');path.write_text(text)
    print('ENCODED_SIGN_SOURCE_READY')


if __name__=='__main__':main()
