"""受控Gaussian弱尾裁剪：Pixel阈值与Tile成员阈值同步变化，输入身份/K/radii不变。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_tail_cutoff';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_tail_cutoff').mkdir()
    helper='''template<int Cut> __device__ __forceinline__ float cutoffAlpha(float encoded,bool done){
  if(done)return 0.f;
  constexpr float floor=(Cut==1?1.25f:Cut==2?1.5f:2.f)/255.f;
  const int bits=min(__float2int_rz(encoded),__float_as_int(.99f));
  return bits<__float_as_int(floor)?0.f:__int_as_float(bits);
}
'''
    for name in ['encoded_sign_pixels.cuh','encoded_sign_diagnosed.cuh']:
        text=(root/name).read_text().replace('encodedSign','cutoff').replace('EncodedSign','Cutoff')
        text=text.replace('template<int Kind,int Step>','template<int Kind,int Step,int Cut>')
        text=text.replace('template<int Mode,int Kind,int Layout>','template<int Mode,int Kind,int Layout,int Cut>')
        text=text.replace('<Kind,Step>','<Kind,Step,Cut>').replace('encodedAlpha(power,T<0.f)','cutoffAlpha<Cut>(power,T<0.f)')
        if name=='encoded_sign_pixels.cuh':text=helper+text
        (root/name.replace('encoded_sign','cutoff')).write_text(text)
    path=root/'preprocess_tasks.cuh';text=path.read_text().replace('template<int ThreadsPerTask>','template<int ThreadsPerTask,int Cut=0>')
    text=text.replace('logf(co.w/(1.0f/255.0f))','logf(co.w/((Cut==0?1.f:Cut==1?1.25f:Cut==2?1.5f:2.f)/255.f))');path.write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "cutoff_pixels.cuh"\n#include "cutoff_diagnosed.cuh"')
    text=text.replace('build_mode<=11','build_mode<=20').replace('build_mode_==10||build_mode_==11','build_mode_==10||build_mode_==11||build_mode_>=18')
    text=text.replace('pixel_mode<=202','pixel_mode<=221').replace('pixel_mode_<=202','pixel_mode_<=221')
    anchor=next(line for line in text.splitlines() if 'else if(build_mode_==10)executePreprocessedTasks' in line)
    calls=[]
    for build,cut in [(18,1),(19,2),(20,3)]:calls.append(anchor.replace('else if(build_mode_==10)',f'else if(build_mode_=={build})').replace('executePreprocessedTasks<32>','executePreprocessedTasks<32,'+str(cut)+'>'))
    text=text.replace(anchor,anchor+'\n'+'\n'.join(calls))
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    calls=[];diagnostics=[]
    for mode,cut in [(201,1),(202,2),(203,3)]:
        calls.append(f'      if(pixel_mode_=={mode+18})renderCutoffPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderCutoffDiagnosedPixels<38,3,1,{cut}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(calls))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostics)+'\n'+anchor).replace('StudentEncodedSignRasterizer','StudentTailCutoffRasterizer');path.write_text(text)
    print('TAIL_CUTOFF_SOURCE_READY')


if __name__=='__main__':main()
