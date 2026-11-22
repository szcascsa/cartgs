"""独立精度候选：log2变换提升至每Gaussian/Tile装载，像素内直接MUFU.EX2。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_exp2_encoded'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_exp2_encoded').mkdir()
    helper = '''__device__ __forceinline__ float liftedExp2Alpha(float power,bool done){
  float exponent;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(exponent):"f"(power));
  const float value=fminf(.99f,exponent);
  return done||value<1.f/255.f?0.f:value;
}
'''
    for name in ['encoded_sign_pixels.cuh', 'encoded_sign_diagnosed.cuh']:
        text = (root / name).read_text().replace('encodedSign', 'exp2Sign').replace('EncodedSign', 'Exp2Sign')
        text = text.replace('encodedAlpha(', 'liftedExp2Alpha(')
        assert text.count('12102203.161561486f') == 6
        text = text.replace('12102203.161561486f', '1.4426950408889634f').replace('1064970947.f', '0.f')
        (root / name.replace('encoded_sign', 'exp2_encoded')).write_text((helper if name.endswith('_pixels.cuh') else '') + text)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "exp2_encoded_pixels.cuh"\n#include "exp2_encoded_diagnosed.cuh"')
    text = text.replace('pixel_mode<=202', 'pixel_mode<=279').replace('pixel_mode_<=202', 'pixel_mode_<=279')
    anchor = next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    call = '      if(pixel_mode_==279)renderExp2SignPixels<38,3,1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);'
    text = text.replace(anchor, anchor + '\n' + call)
    anchor = '    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    diagnostic = '    if(pixel_mode_==279)renderExp2SignDiagnosedPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));'
    assert text.count(anchor) == 1
    text = text.replace(anchor, diagnostic + '\n' + anchor)
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentExp2EncodedRasterizer'))
    print('EXP2_ENCODED_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
