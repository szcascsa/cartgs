"""计时外插桩使用同一候选表达式，统计实际参与/交互，并独立写出用于逐位自检。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/DEV_CONTROLLER06.json').read_text())['state']=='complete'
    snapshot=HERE/'development_v5';snapshot.mkdir();shutil.copytree(HERE/'source',snapshot/'source')
    shutil.copy2(HERE/'build/mycartgs_footprint_ninehour_v1.so',snapshot/'mycartgs_footprint_ninehour_v1.so')
    shutil.copy2(HERE/'render_runtime.py',snapshot/'render_runtime.py')
    text=(HERE/'source/folded_pixels.cuh').read_text()
    for old,new in [('boundedExp','debugBoundedExp'),('FoldedGaussian','DebugFoldedGaussian'),('foldGaussian','debugFoldGaussian'),
                    ('blendFolded','debugBlendFolded'),('foldedPair','debugFoldedPair'),('renderFoldedPixels','renderDiagnosedPixels')]:text=text.replace(old,new)
    text=text.replace('float power,float3 color,bool& done,float& T,float* C)',
        'float power,float3 color,bool& done,float& T,float* C,unsigned id,unsigned* flags,unsigned long long* visits)')
    text=text.replace('  T=next;', '  T=next;')
    needle='''  constexpr float Stop='''
    assert needle in text;text=text.replace(needle,'  atomicOr(flags+id,1u);++*visits;\n'+needle)
    text=text.replace('float x,float y,bool* done,float* T,float C[2][3])',
        'float x,float y,bool* done,float* T,float C[2][3],unsigned id,unsigned* flags,unsigned long long* visits)')
    text=text.replace('done[0],T[0],C[0]);','done[0],T[0],C[0],id,flags,visits);').replace('done[1],T[1],C[1]);','done[1],T[1],C[1],id,flags,visits);')
    text=text.replace('const GraphInputs* inputs) {','const GraphInputs* inputs,float* debug_image,unsigned* flags,unsigned long long* interactions) {')
    text=text.replace('float* out=inputs->image;','float* out=debug_image;unsigned long long visits=0;')
    text=text.replace('rgb[j],lx,ly,done,T,C);','rgb[j],lx,ly,done,T,C,lists[tile*capacity+j],flags,&visits);')
    text=text.replace('{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C);',
        '{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C,id,flags,&visits);')
    text=text.replace('compositeScalingPair<9>','debugOriginalPair')
    text=text.replace('pix,done,T,C,-CUDART_INF_F);','pix,done,T,C,-CUDART_INF_F,id,flags,&visits);')
    # 主短列表原fallback的id在局部声明；全Tilefallback和overflow本来已经有id。
    text=text.replace('const float4 a=abc[j];const float2 b=ef[j];','const int id=lists[tile*capacity+j];const float4 a=abc[j];const float2 b=ef[j];')
    anchor='''  #pragma unroll
  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out'''
    assert anchor in text;text=text.replace(anchor,'  if(visits)atomicAdd(interactions,visits);\n'+anchor)
    helper='''__device__ __forceinline__ void debugOriginalPair(float2 point,float4 co,float3 color,float2* pix,
    bool* done,float* T,float C[2][3],float bound,unsigned id,unsigned* flags,unsigned long long* visits){
  #pragma unroll
  for(int p=0;p<2;++p){const float before=T[p];compositeScaling<9>(point,co,color,pix[p],done[p],T[p],C[p],bound);
    if(T[p]!=before){atomicOr(flags+id,1u);++*visits;}}
}
'''
    (HERE/'source/diagnosed_pixels.cuh').write_text(helper+text)
    path=HERE/'source/graph_main.cu';s=path.read_text().replace('#include "tensor_pixels.cuh"','#include "tensor_pixels.cuh"\n#include "diagnosed_pixels.cuh"')
    method='''  std::vector<Tensor> diagnose()const {
    TORCH_CHECK(!plans.empty()&&pixel_mode_>=55&&pixel_mode_<=58,"Diagnostic modes37..40 require prior current-frame render");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==owner,"Use one graph renderer per device/stream");
    auto found=std::max_element(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a->last_used<b->last_used;});
    const auto& p=**found;const int n=p.capacity,W=p.W,H=p.H,tiles=((W+15)/16)*((H+15)/16),words=(n+31)/32,groups=(words+31)/32;
    char* geom=reinterpret_cast<char*>(p.geometry.data_ptr());
    using CudaRasterizer::obtain;float* depths;bool* clamped;int* radii;float2* xy;float* covariance;float4* conic;float* rgb;uint32_t* touched;
    obtain(geom,depths,n,128);obtain(geom,clamped,n*3,128);obtain(geom,radii,n,128);obtain(geom,xy,n,128);
    obtain(geom,covariance,n*6,128);obtain(geom,conic,n,128);obtain(geom,rgb,n*3,128);obtain(geom,touched,n,128);
    char* sorting=reinterpret_cast<char*>(p.sort.data_ptr());uint32_t *ki,*ko,*ii,*sorted;
    obtain(sorting,ki,n,128);obtain(sorting,ko,n,128);obtain(sorting,ii,n,128);obtain(sorting,sorted,n,128);
    auto image=torch::empty({3,H,W},parameters.options().dtype(at::kFloat));
    auto flags=torch::zeros({n},parameters.options().dtype(at::kInt));auto interactions=torch::zeros({1},parameters.options().dtype(at::kLong));
    auto bits=reinterpret_cast<uint32_t*>(p.bits.data_ptr());auto directory=reinterpret_cast<uint32_t*>(p.directory.data_ptr());
    auto lists=reinterpret_cast<uint32_t*>(p.lists.data_ptr());auto counts=reinterpret_cast<uint32_t*>(p.counts.data_ptr());
    auto inputs=reinterpret_cast<const GraphInputs*>(parameters.data_ptr());auto stream=c10::cuda::getCurrentCUDAStream();dim3 grid((W+15)/16,(H+15)/16);
'''
    for m in range(37,41):method+=f'''    if(pixel_mode_=={18+m})renderDiagnosedPixels<{m}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));
'''
    method+='''    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};
  }
'''
    assert '  std::vector<Tensor> snapshot()const {' in s;s=s.replace('  std::vector<Tensor> snapshot()const {',method+'  std::vector<Tensor> snapshot()const {')
    needle='.def("snapshot",&NineHourRasterizer::snapshot)';assert needle in s;s=s.replace(needle,'.def("diagnose",&NineHourRasterizer::diagnose)'+needle)
    path.write_text(s)
    print('DIAGNOSTIC_ADDED_OUTSIDE_TIMING')


if __name__=='__main__':main()
