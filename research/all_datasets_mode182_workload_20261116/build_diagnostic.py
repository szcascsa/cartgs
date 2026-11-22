"""只在独立研究目录构建诊断扩展，复用发布版投影对象，不修改发布源码或二进制。"""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT.parent/'low_ratio_footprint_tc_9h_20261112_0011/student_resume_20261114_2250'
SOURCE = RELEASE/'source_joint_release/raster'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


HELPERS = r'''
// Counter slots: scheduled pair slots, valid active tests, effective blends, fallback tests.
__device__ __forceinline__ void workloadBlend(float power,float3 color,float& T,float* C,
    bool inside,unsigned long long* work){
  ++work[0];if(inside&&T>=0.f)++work[1];
  const float alpha=encodedAlpha(power,T<0.f);
  const float weight=alpha*T;
  const float next=T-weight;
  C[0]=fmaf(color.x,weight,C[0]);C[1]=fmaf(color.y,weight,C[1]);C[2]=fmaf(color.z,weight,C[2]);
  if(inside&&weight>0.f)++work[2];
  T=__int_as_float(__float_as_int(next)|(next<.01f?int(0x80000000u):0));
}

template<int Kind,int Step> __device__ __forceinline__ void workloadPair(FoldedGaussian f,float3 color,
    float x,float y,float* T,float C[2][3],const bool* inside,unsigned long long* work){
  const float power=fmaf(f.a.x,x*x,fmaf(f.a.z,y*y,fmaf(f.a.y,x*y,fmaf(f.a.w,x,fmaf(f.b.x,y,f.b.y)))));
  const float next=power+float(Step)*(f.a.y*x+f.a.z*(2.f*y+float(Step))+f.b.x);
  workloadBlend(power,color,T[0],C[0],inside[0],work);workloadBlend(next,color,T[1],C[1],inside[1],work);
}

__device__ __forceinline__ void workloadOriginal(float2 point,float4 co,float3 color,float2 pix,
    bool& done,float& T,float* C,bool inside,unsigned long long* work){
  ++work[0];if(done)return;
  if(inside){++work[1];++work[3];}
  float2 d={point.x-pix.x,point.y-pix.y};
  float power=-0.5f*(co.x*d.x*d.x+co.z*d.y*d.y)-co.y*d.x*d.y;
  if(power>0.0f)return;
  float alpha=min(0.99f,co.w*exp(power));if(alpha<1.0f/255.0f)return;
  float test_T=T*(1-alpha);if(test_T<0.0001f){done=true;return;}
  C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;
  T=test_T;if(inside)++work[2];
}

__device__ __forceinline__ void workloadFallback(float2 xy,float4 conic,float3 rgb,float2* pix,
    float* T,float C[2][3],const bool* inside,unsigned long long* work){
  bool done[2]={T[0]<0.f,T[1]<0.f};T[0]=fabsf(T[0]);T[1]=fabsf(T[1]);
  #pragma unroll
  for(int p=0;p<2;++p)workloadOriginal(xy,conic,rgb,pix[p],done[p],T[p],C[p],inside[p],work);
  if(done[0])T[0]=-T[0];if(done[1])T[1]=-T[1];
}
'''

END_REDUCTION = r'''
  __shared__ unsigned long long reduction[6][128];
  for(int c=0;c<4;++c)reduction[c][tid]=work[c];
  reduction[4][tid]=(inside[0]&&fabsf(T[0])<1.f)+(inside[1]&&fabsf(T[1])<1.f);
  reduction[5][tid]=(inside[0]&&T[0]<0.f)+(inside[1]&&T[1]<0.f);
  __syncthreads();
  if(tid<6){unsigned long long sum=0;for(int i=0;i<128;++i)sum+=reduction[tid][i];
    constexpr int col[6]={2,3,4,5,8,9};metrics[tile*12+col[tid]]=sum;}
  if(!tid){
    const int valid_w=min(16,W-int(blockIdx.x)*16),valid_h=min(16,H-int(blockIdx.y)*16);
    metrics[tile*12+0]=count;metrics[tile*12+1]=int64_t(count)*valid_w*valid_h;
    metrics[tile*12+6]=count>capacity;metrics[tile*12+7]=count>0;
    metrics[tile*12+10]=int64_t(count)*count;metrics[tile*12+11]=count;
  }
'''


def main():
    source, build = ROOT/'source', ROOT/'build'
    source.mkdir(exist_ok=True)
    build.mkdir(exist_ok=True)
    original = (SOURCE/'encoded_sign_pixels.cuh').read_text()
    kernel = original[original.index('template<int Mode,int Kind,int Layout> __global__ void'):]
    kernel = kernel.replace('renderEncodedSignPixels(', 'renderWorkloadPixels(')
    kernel = kernel.replace('const GraphInputs* inputs) {', 'const GraphInputs* inputs,float* image,int64_t* metrics) {')
    kernel = kernel.replace('float* out=inputs->image;', 'float* out=image;unsigned long long work[4]={0,0,0,0};')
    kernel = kernel.replace('    return;\n  }\n  const int count=counts[tile];',
        '    if(!tid)for(int c=0;c<12;++c)metrics[tile*12+c]=0;\n    return;\n  }\n  const int count=counts[tile];')
    kernel = kernel.replace('encodedSignFallback(', 'workloadFallback(').replace('pix,T,C);', 'pix,T,C,inside,work);')
    kernel = kernel.replace('encodedSignPair<Kind,Step>(', 'workloadPair<Kind,Step>(')
    kernel = kernel.replace('lx,ly,T,C);', 'lx,ly,T,C,inside,work);')
    point = '  #pragma unroll\n  for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)out[ch*H*W+(y+Step*p)*W+x]=C[p][ch]+fabsf(T[p])*bg[ch];'
    assert kernel.count(point) == 1
    kernel = kernel.replace(point, END_REDUCTION+'\n'+point)
    (source/'workload_pixels.cuh').write_text(HELPERS+'\n'+kernel)
    graph = (SOURCE/'graph_main.cu').read_text()
    graph = re.sub(r'#include "([^"]+)"', lambda m: '#include "'+str((SOURCE/m[1]).resolve())+'"', graph)
    graph = graph.replace('#include <list>', '#include "workload_pixels.cuh"\n#include <list>')
    start = graph.index('  std::vector<Tensor> diagnose()const {')
    header_end = graph.index('    auto image=torch::empty(', start)
    header = graph[start:header_end].replace('diagnose()const', 'workload()const')
    header = header.replace('!plans.empty()&&pixel_mode_>=55&&pixel_mode_<=320', '!plans.empty()&&pixel_mode_==200')
    method = header+r'''
    auto image=torch::empty({3,H,W},parameters.options().dtype(at::kFloat));
    auto metrics=torch::empty({tiles,12},parameters.options().dtype(at::kLong));
    auto bits=reinterpret_cast<uint32_t*>(p.bits.data_ptr());auto directory=reinterpret_cast<uint32_t*>(p.directory.data_ptr());
    auto lists=reinterpret_cast<uint32_t*>(p.lists.data_ptr());auto counts=reinterpret_cast<uint32_t*>(p.counts.data_ptr());
    auto inputs=reinterpret_cast<const GraphInputs*>(parameters.data_ptr());auto stream=c10::cuda::getCurrentCUDAStream();dim3 grid((W+15)/16,(H+15)/16);
    renderWorkloadPixels<38,3,1><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),metrics.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto total=metrics.sum(0);auto maximum=std::get<0>(metrics.select(1,11).max(0));
    return {image,total,maximum};
  }
'''
    graph = graph[:start]+method+graph[start:]
    graph = graph.replace('(m,"PendingRasterizer")', '(m,"WorkloadRasterizer",pybind11::module_local())')
    graph = graph.replace('.def("diagnose",&StudentJointReleaseRasterizer::diagnose)',
        '.def("workload",&StudentJointReleaseRasterizer::workload).def("diagnose",&StudentJointReleaseRasterizer::diagnose)')
    (source/'graph_main.cu').write_text(graph)
    (source/'bindings.cpp').write_text('#include <torch/extension.h>\nvoid register_pending_raster(pybind11::module_&);\n'
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){register_pending_raster(m);}\n')
    linked = RELEASE/'build_joint_final/graph_forward.cuda.o'
    protected = {str(path): sha(path) for path in [SOURCE/'graph_main.cu', SOURCE/'encoded_sign_pixels.cuh',
        SOURCE/'scaling_pixels.cuh', SOURCE/'encoded_sign_diagnosed.cuh', linked,
        RELEASE/'build_joint_release/mycartgs_student_joint_release_20261114_v1.so']}
    (ROOT/'BUILD_INPUTS.json').write_text(json.dumps(protected, indent=2)+'\n')
    os.environ.update(CUDA_HOME='/usr/local/cuda-11.8', TORCH_CUDA_ARCH_LIST='8.6', MAX_JOBS='2')
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    from torch.utils.cpp_extension import load
    module = load(name='mycartgs_mode182_workload_v1', sources=[str(source/'bindings.cpp'),str(source/'graph_main.cu')],
        extra_include_paths=[str(SOURCE/'reference/cuda_rasterizer')], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'], extra_ldflags=[str(linked)], build_directory=str(build), verbose=True)
    assert protected == {path: sha(Path(path)) for path in protected}
    result = dict(passed=True,binary=module.__file__,sha256=sha(Path(module.__file__)),
        original_files_unchanged=True,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source.iterdir()})
    (ROOT/'BUILD_VALIDATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print('DIAGNOSTIC_BUILD_PASS', json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
