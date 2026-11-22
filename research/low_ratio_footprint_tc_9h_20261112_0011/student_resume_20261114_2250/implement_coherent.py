"""保持一次Tile共享加载，独立比较warp像素形状及T-w递推。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
MODES={151:(3,1),152:(3,2),153:(3,3),154:(5,0),155:(5,1),156:(1,1)}


def main():
    root=HERE/'source_coherent'
    shutil.copytree(HERE/'source_branchless',root)
    (HERE/'build_coherent').mkdir()
    for filename in ['branchless_pixels.cuh','branchless_diagnosed.cuh']:
        text=(root/filename).read_text()
        if filename=='branchless_pixels.cuh':text=text[text.index('template<int Kind>'):]
        text=text.replace('branchlessBlend','coherentBlend').replace('branchlessPair','coherentPair')
        text=text.replace('diagnosedBranchless','diagnosedCoherent').replace('renderBranchless','renderCoherent')
        text=text.replace('template<int Kind>', 'template<int Kind,int Step>')
        text=text.replace('template<int Mode,int Kind>', 'template<int Mode,int Kind,int Layout>')
        text=text.replace('coherentBlend<Kind>', 'coherentBlend<Kind,Step>')
        text=text.replace('diagnosedCoherentBlend<Kind>', 'diagnosedCoherentBlend<Kind,Step>')
        text=text.replace('coherentPair<Kind>', 'coherentPair<Kind,Step>')
        text=text.replace('diagnosedCoherentPair<Kind>', 'diagnosedCoherentPair<Kind,Step>')
        text=text.replace('const float next=power+8.f*(f.a.y*x+f.a.z*(2.f*y+8.f)+f.b.x);',
            'const float next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);')
        text=text.replace('float next=T*(1.f-alpha),weight=alpha*T;',
            'float weight=alpha*T;float next=Kind==5?T-weight:T*(1.f-alpha);')
        old='  const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;'
        new='''  constexpr int Step=Layout==0?8:1;
  const int lane=tid&31,warp=tid>>5;
  const int px=Layout==0?int(threadIdx.x):Layout==1?(warp&1)*8+(lane&7):Layout==2?(lane&15):warp*4+(lane&3);
  const int py=Layout==0?int(threadIdx.y):Layout==1?(warp>>1)*8+(lane>>3)*2:Layout==2?warp*4+(lane>>4)*2:(lane>>2)*2;
  const int x=blockIdx.x*16+px,y=blockIdx.y*16+py;'''
        assert old in text
        text=text.replace(old,new).replace('threadIdx.x-7.5f','px-7.5f').replace('threadIdx.y-7.5f','py-7.5f')
        text=text.replace('y+8','y+Step').replace('8*p','Step*p')
        (root/filename.replace('branchless','coherent')).write_text(text)
    path=root/'graph_main.cu'
    text=path.read_text().replace('#include "branchless_diagnosed.cuh"',
        '#include "branchless_diagnosed.cuh"\n#include "coherent_pixels.cuh"\n#include "coherent_diagnosed.cuh"')
    text=text.replace('pixel_mode<=162','pixel_mode<=174').replace('pixel_mode_<=162','pixel_mode_<=174')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==162)renderBranchlessPixels' in line)
    launches=[];diagnostic=[]
    for mode,(kind,layout) in MODES.items():
        launches.append(f'      if(pixel_mode_=={mode+18})renderCoherentPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostic.append(f'    if(pixel_mode_=={mode+18})renderCoherentDiagnosedPixels<38,{kind},{layout}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(launches))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostic)+'\n'+anchor).replace('StudentBranchlessRasterizer','StudentCoherentRasterizer')
    path.write_text(text)
    print('COHERENT_SOURCE_READY')


if __name__=='__main__':main()
