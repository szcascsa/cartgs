"""把颜色补全移到像素退出后，热循环只保留原有透射率早停判断。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
STOPS={61:.02,62:.03,63:.04,64:.05,65:.075,66:.1,67:.15,68:.2,69:.35}


def main():
    root=HERE/'source_completion';shutil.copytree(HERE.parent/'source',root);(HERE/'build_completion').mkdir()
    for name in ['folded_pixels.cuh','diagnosed_pixels.cuh']:
        path=root/name;text=path.read_text()
        text=text.replace('Mode>=37&&Mode<=39?1:', 'Mode>=37&&Mode<=39||Mode>=61?1:')
        stop=':'.join(f'Mode=={m}?{value:g}f' for m,value in STOPS.items())+':0.f'
        text=text.replace('Mode==18?.003f:0.f;', 'Mode==18?.003f:'+stop+';')
        text=text.replace('  const int count=counts[tile];', '  bool completion_allowed=false;\n  const int count=counts[tile];')
        anchor='    #pragma unroll 32\n    for(int j=0;j<count;++j)'
        assert text.count(anchor)==1
        text=text.replace(anchor,'    if(Mode>=61)completion_allowed=true;\n'+anchor)
        anchor='  #pragma unroll\n  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out'
        assert text.count(anchor)==1
        completion='''  if(Mode>=61&&completion_allowed){
    constexpr float cutoff='''+stop+''';
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p]&&done[p]&&T[p]<cutoff){
      const float inverse=1.f/(1.f-T[p]);C[p][0]*=inverse;C[p][1]*=inverse;C[p][2]*=inverse;T[p]=0.f;
    }
  }
'''
        text=text.replace(anchor,completion+anchor)
        path.write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('pixel_mode<=58','pixel_mode<=87')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[f'      if(pixel_mode_=={m+18})renderFoldedPixels<{m}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);' for m in STOPS]
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines))
    text=text.replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=87')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    lines=[f'    if(pixel_mode_=={m+18})renderDiagnosedPixels<{m}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));' for m in STOPS]
    assert text.count(anchor)==1
    text=text.replace(anchor,'\n'.join(lines)+'\n'+anchor).replace('NineHourRasterizer','StudentCompletionRasterizer')
    path.write_text(text)
    print('STUDENT_COMPLETION_SOURCE_READY')


if __name__=='__main__':main()
