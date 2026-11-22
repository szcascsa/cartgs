"""bit-exp输入线性变换提升和Horner多项式；所有近似独立开关。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_encoded';shutil.copytree(HERE/'source_coherent',root);(HERE/'build_encoded').mkdir()
    helper=(HERE/'encoded_power.cuh').read_text()
    diagnostic=helper[helper.index('template<int Kind'):].replace('encodedBlend','diagnosedEncodedBlend').replace('encodedPair','diagnosedEncodedPair').replace('FoldedGaussian','DebugFoldedGaussian')
    diagnostic=diagnostic.replace('float* C){','float* C,unsigned id,unsigned* flags,unsigned long long* visits){')
    diagnostic=diagnostic.replace('float C[2][3]){','float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits){')
    diagnostic=diagnostic.replace('C[0]);diagnosedEncodedBlend','C[0],id,flags,visits);diagnosedEncodedBlend').replace('T[1],C[1]);','T[1],C[1],id,flags,visits);')
    diagnostic=diagnostic.replace('  T=T-weight;', '  if(weight>0.f){atomicOr(flags+id,1u);++*visits;}\n  T=T-weight;')
    scale='''if constexpr(Kind<=3){
        f.a.x*=12102203.161561486f;f.a.y*=12102203.161561486f;f.a.z*=12102203.161561486f;f.a.w*=12102203.161561486f;
        f.b.x*=12102203.161561486f;f.b.y=fmaf(f.b.y,12102203.161561486f,1064970947.f);
      }'''
    for filename,debug in [('coherent_pixels.cuh',False),('coherent_diagnosed.cuh',True)]:
        text=(root/filename).read_text();text=text[text.index('template<int Mode,int Kind,int Layout>'):]
        text=text.replace('renderCoherent','renderEncoded').replace('coherentPair<Kind,Step>','encodedPair<Kind,Step>').replace('diagnosedCoherentPair<Kind,Step>','diagnosedEncodedPair<Kind,Step>')
        assert text.count('abc[j]=f.a')==1
        text=text.replace('abc[j]=f.a',scale+'\n      abc[j]=f.a')
        (root/filename.replace('coherent','encoded')).write_text((diagnostic if debug else helper)+'\n'+text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "coherent_diagnosed.cuh"',
        '#include "coherent_diagnosed.cuh"\n#include "encoded_pixels.cuh"\n#include "encoded_diagnosed.cuh"')
    text=text.replace('pixel_mode<=174','pixel_mode<=192').replace('pixel_mode_<=174','pixel_mode_<=192')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==174)renderCoherentPixels' in line)
    launches=[];diagnostics=[]
    for mode,kind in [(171,1),(172,2),(173,3),(174,4)]:
        launches.append(f'      if(pixel_mode_=={mode+18})renderEncodedPixels<38,{kind},1><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderEncodedDiagnosedPixels<38,{kind},1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(launches))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostics)+'\n'+anchor).replace('StudentCoherentRasterizer','StudentEncodedRasterizer');path.write_text(text)
    print('ENCODED_SOURCE_READY')


if __name__=='__main__':main()
