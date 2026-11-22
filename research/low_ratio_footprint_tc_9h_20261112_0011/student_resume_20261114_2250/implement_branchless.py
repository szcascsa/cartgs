"""合并bit-exp拒绝、尝试predication，独立保留严格min-T与近似省略两种路径。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_branchless';shutil.copytree(HERE.parent/'source',root);(HERE/'build_branchless').mkdir()
    text=(root/'folded_pixels.cuh').read_text();kernel=text[text.index('template<int Mode> __global__ void renderFoldedPixels'):]
    kernel=kernel.replace('renderFoldedPixels','renderBranchlessPixels').replace('template<int Mode>','template<int Mode,int Kind>')
    kernel=kernel.replace('foldedPair<Mode>', 'branchlessPair<Kind>')
    (root/'branchless_pixels.cuh').write_text((HERE/'branchless_blend.cuh').read_text()+'\n'+kernel)
    helpers=(HERE/'branchless_blend.cuh').read_text()
    helpers=helpers[helpers.index('template<int Kind> __device__'):]
    helpers=helpers.replace('branchlessBlend','diagnosedBranchlessBlend').replace('branchlessPair','diagnosedBranchlessPair')
    helpers=helpers.replace('float power,float3 color,bool& done,float& T,float* C)', 'float power,float3 color,bool& done,float& T,float* C,unsigned id,unsigned* flags,unsigned long long* visits)')
    helpers=helpers.replace('FoldedGaussian f','DebugFoldedGaussian f')
    helpers=helpers.replace('float x,float y,bool* done,float* T,float C[2][3])', 'float x,float y,bool* done,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits)')
    helpers=helpers.replace('C[0]);diagnosedBranchlessBlend', 'C[0],id,flags,visits);diagnosedBranchlessBlend').replace('T[1],C[1]);','T[1],C[1],id,flags,visits);')
    helpers=helpers.replace('    T=next;if(T<.01f)done=true;', '    if(weight>0.f){atomicOr(flags+id,1u);++*visits;}\n    T=next;if(T<.01f)done=true;')
    diag=(root/'diagnosed_pixels.cuh').read_text();diag=diag[diag.index('template<int Mode> __global__ void renderDiagnosedPixels'):]
    diag=diag.replace('template<int Mode>','template<int Mode,int Kind>').replace('renderDiagnosedPixels','renderBranchlessDiagnosedPixels')
    diag=diag.replace('debugFoldedPair<Mode>','diagnosedBranchlessPair<Kind>')
    (root/'branchless_diagnosed.cuh').write_text(helpers+'\n'+diag)
    # 独立测试合并Alpha拒绝是否严格等价。
    text='''#include <cuda_runtime.h>
__global__ void checkMergedAlpha(int n,const float* powers,float* old_alpha,float* new_alpha){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float p=powers[i],a=0.f;
  if(!(p<-5.541263545158426f)){a=fminf(.99f,boundedExp<1>(p));if(a<1.f/255.f)a=0.f;}
  old_alpha[i]=a;new_alpha[i]=mergedFoldedAlpha(p,false);
}
std::vector<torch::Tensor> alphaParity(const torch::Tensor& powers){
  TORCH_CHECK(powers.is_cuda()&&powers.scalar_type()==at::kFloat&&powers.is_contiguous(),"Alpha parity FP32 CUDA");
  auto a=torch::empty_like(powers),b=torch::empty_like(powers);
  checkMergedAlpha<<<(powers.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(powers.numel(),powers.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {a,b};
}
'''
    (root/'alpha_parity.cuh').write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "branchless_pixels.cuh"\n#include "branchless_diagnosed.cuh"\n#include "alpha_parity.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=162')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,kind in [(141,1),(142,2),(143,3),(144,4)]:
        lines.append(f'      if(pixel_mode_=={mode+18})renderBranchlessPixels<38,{kind}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnosed.append(f'    if(pixel_mode_=={mode+18})renderBranchlessDiagnosedPixels<38,{kind}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=162')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentBranchlessRasterizer')
    anchor='PYBIND11_MODULE(TORCH_EXTENSION_NAME'
    i=text.index(anchor);brace=text.index('{',i)
    text=text[:brace+1]+'\n  m.def("_alpha_parity",&alphaParity);'+text[brace+1:]
    path.write_text(text)
    print('STUDENT_BRANCHLESS_SOURCE_READY')


if __name__=='__main__':main()
