"""T符号状态与原Tile内布局比较；不改成员、排序、预处理与发布。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
MODES={161:(1,0),162:(2,0),163:(1,1),164:(2,1),165:(3,0),166:(3,1)}


def main():
    root=HERE/'source_sign';shutil.copytree(HERE/'source_coherent',root);(HERE/'build_sign').mkdir()
    helper=(HERE/'sign_blend.cuh').read_text()
    diagnostic=helper.replace('signBlend','diagnosedSignBlend').replace('signPair','diagnosedSignPair')
    diagnostic=diagnostic.replace('float* C){','float* C,unsigned id,unsigned* flags,unsigned long long* visits){')
    diagnostic=diagnostic.replace('float C[2][3]){','float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits){')
    diagnostic=diagnostic.replace('FoldedGaussian','DebugFoldedGaussian')
    diagnostic=diagnostic[:diagnostic.index('__device__ __forceinline__ void signFallback')]
    diagnostic=diagnostic.replace('C[0]);diagnosedSignBlend','C[0],id,flags,visits);diagnosedSignBlend').replace('T[1],C[1]);','T[1],C[1],id,flags,visits);')
    diagnostic=diagnostic.replace('  T=__int_as_float','  if(weight>0.f){atomicOr(flags+id,1u);++*visits;}\n  T=__int_as_float')
    for filename,is_debug in [('coherent_pixels.cuh',False),('coherent_diagnosed.cuh',True)]:
        text=(root/filename).read_text();text=text[text.index('template<int Mode,int Kind,int Layout>'):]
        text=text.replace('renderCoherent','renderSign')
        text=text.replace('float T[2]={1.f,1.f}', 'float T[2]={inside[0]?1.f:-1.f,inside[1]?1.f:-1.f}')
        text=text.replace('done[0]&&done[1]','T[0]<0.f&&T[1]<0.f')
        text=text.replace('if(T[0]<0.f&&T[1]<0.f)continue;','if constexpr(Kind<3){if(T[0]<0.f&&T[1]<0.f)continue;}')
        text=text.replace('coherentPair<Kind,Step>','signPair<Kind,Step>').replace('diagnosedCoherentPair<Kind,Step>','diagnosedSignPair<Kind,Step>')
        text=text.replace('lx,ly,done,T,C','lx,ly,T,C')
        if is_debug:
            # 慢路径保留原诊断函数，局部转换符号状态。
            # 安全正常路径无需done；溢出/unsafe采用原版并恢复符号。
            lines=[]
            for line in text.splitlines():
                if 'debugScalingPair<9>' in line or 'compositeScalingPair<9>' in line:
                    line='        done[0]=T[0]<0.f;done[1]=T[1]<0.f;T[0]=fabsf(T[0]);T[1]=fabsf(T[1]);\n'+line+'\n        if(done[0])T[0]=-T[0];if(done[1])T[1]=-T[1];'
                lines.append(line)
            text='\n'.join(lines)
        else:
            text=text.replace('compositeScalingPair<9>','signFallback').replace('pix,done,T,C,-CUDART_INF_F','pix,T,C')
        text=text.replace('C[p][ch]+T[p]*bg[ch]','C[p][ch]+fabsf(T[p])*bg[ch]')
        (root/filename.replace('coherent','sign')).write_text((diagnostic if is_debug else helper)+'\n'+text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "coherent_diagnosed.cuh"',
        '#include "coherent_diagnosed.cuh"\n#include "sign_pixels.cuh"\n#include "sign_diagnosed.cuh"')
    text=text.replace('pixel_mode<=174','pixel_mode<=184').replace('pixel_mode_<=174','pixel_mode_<=184')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==174)renderCoherentPixels' in line)
    launches=[];diagnostics=[]
    for mode,(kind,layout) in MODES.items():
        launches.append(f'      if(pixel_mode_=={mode+18})renderSignPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderSignDiagnosedPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(launches))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostics)+'\n'+anchor).replace('StudentCoherentRasterizer','StudentSignRasterizer');path.write_text(text)
    print('SIGN_SOURCE_READY')


if __name__=='__main__':main()
