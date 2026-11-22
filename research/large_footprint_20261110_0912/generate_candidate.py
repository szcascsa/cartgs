"""仅在本轮源码副本注入独立成员构建候选，不改原版或上一轮二进制。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/'source/graph_main.cu';text=path.read_text(encoding='utf-8')
assert 'FootprintGraphRasterizer' not in text
text=text.replace('#include "graph_kernels.cuh"','#include "graph_kernels.cuh"\n#include "footprint_build.cuh"',1)
text=text.replace('CapacityGraphRasterizer','FootprintGraphRasterizer')
text=text.replace('int max_plans=64;', 'int max_plans=64;int build_mode_=0;Tensor pool_metadata;')
needle='FootprintGraphRasterizer(int count=64,bool ordered=false,bool device_clear=false):max_plans(count),ordered_coverage_(ordered),device_clear_(device_clear)'
assert text.count(needle)==1
text=text.replace(needle,'FootprintGraphRasterizer(int count=64,bool ordered=true,bool device_clear=true,int build_mode=0):max_plans(count),build_mode_(build_mode),ordered_coverage_(ordered),device_clear_(device_clear)')
text=text.replace('TORCH_CHECK(count>0&&count<=512,"Graph plan count must be in 1..512");',
    'TORCH_CHECK(count>0&&count<=512,"Graph plan count must be in 1..512");TORCH_CHECK(build_mode>=0&&build_mode<=7&&ordered,"Footprint build mode and exact coverage");',1)
text=text.replace('int64_t(capacity)*128+8192+', 'int64_t(capacity)*148+8192+',1)
text=text.replace('int64_t n=parameters.defined()?parameters.storage().nbytes():0;',
    'int64_t n=parameters.defined()?parameters.storage().nbytes():0;\n    if(pool_metadata.defined())n+=pool_metadata.storage().nbytes();',1)
text=text.replace('pool_lists=Tensor();pool_counts=Tensor();pool_tasks=Tensor();',
    'pool_lists=Tensor();pool_counts=Tensor();pool_tasks=Tensor();pool_metadata=Tensor();',1)
text=text.replace('pool_tasks=buffer(int64_t(capacity)*((tiles+255)/256)*8);',
    'pool_tasks=buffer(int64_t(capacity)*((tiles+255)/256)*8);pool_metadata=buffer(int64_t(capacity)*20);',1)
begin=text.index('    if(device_clear_)clearGraphBits')
end=text.index('    if(variant==5)renderMultiPixel',begin)
original=text[begin:end]
control_begin=original.index('    if(ordered_coverage_)')
control_end=original.index('    compactTileBits')
control=original[control_begin:control_end]
new='''    if(build_mode_==7) {
      auto rectangles=reinterpret_cast<uint4*>(pool_metadata.data_ptr());
      auto limits=reinterpret_cast<float*>(rectangles+capacity);
      prepareFootprintMetadata<<<(capacity+127)/128,128,0,capture>>>(capacity,xy,conic,radii,sorted,key_out,touched,
          rectangles,limits,bits+int64_t(tiles)*words,grid);
      buildCompactedTiles<<<(tiles+3)/4,128,0,capture>>>(capacity,sorted,xy,conic,rectangles,limits,
          bits,words,directory,groups,lists,counts,list_capacity,grid);
    } else {
      if(build_mode_!=6) {
        if(device_clear_)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);
        else C10_CUDA_CHECK(cudaMemsetAsync(bits,0,bit_bytes,capture));
      }
      if(build_mode_==0) {
'''+control+'''      }
      else if(build_mode_==1)buildGaussianFootprint<128,1><<<capacity,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==2)buildGaussianFootprint<64,1><<<capacity,64,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==3)buildGaussianFootprint<256,1><<<capacity,256,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==4)buildGaussianFootprint<128,2><<<capacity*2,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==5)buildGaussianFootprint<128,4><<<capacity*4,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      else if(build_mode_==6)buildTileWords<<<tiles,128,0,capture>>>(capacity,sorted,xy,conic,radii,bits,words,grid);
      compactTileBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity,capacity,key_out,touched);
    }
'''
text=text[:begin]+new+text[end:]
needle='  std::vector<int64_t> stats()const'
debug='''  std::vector<Tensor> snapshot()const {
    TORCH_CHECK(!plans.empty(),"No rendered plan");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==owner,"Use one graph renderer per device/stream");
    auto found=std::max_element(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a->last_used<b->last_used;});
    const auto& p=**found;const int n=p.capacity,tiles=((p.W+15)/16)*((p.H+15)/16),words=(n+31)/32,groups=(words+31)/32;
    auto take=[&](void* data,std::vector<int64_t> shape){return torch::from_blob(data,shape,p.bits.options().dtype(at::kInt)).clone();};
    char* sorting=reinterpret_cast<char*>(p.sort.data_ptr());uint32_t *ki,*ko,*ii,*sorted;uint64_t *mi,*mo;
    CudaRasterizer::obtain(sorting,ki,n,128);CudaRasterizer::obtain(sorting,ko,n,128);
    CudaRasterizer::obtain(sorting,ii,n,128);CudaRasterizer::obtain(sorting,sorted,n,128);
    return {take(p.bits.data_ptr(),{tiles,words}),take(p.directory.data_ptr(),{tiles,groups}),
        take(p.lists.data_ptr(),{tiles,p.list_capacity}),take(p.counts.data_ptr(),{tiles}),take(sorted,{n}),take(ko,{n})};
  }
'''
assert text.count(needle)==1;text=text.replace(needle,debug+needle)
text=text.replace('m,"GraphRasterizer").def(pybind11::init<int,bool,bool>(),pybind11::arg("max_plans")=64,pybind11::arg("ordered_coverage")=false,pybind11::arg("device_clear")=false)',
    'm,"FootprintRasterizer").def(pybind11::init<int,bool,bool,int>(),pybind11::arg("max_plans")=512,pybind11::arg("ordered_coverage")=true,pybind11::arg("device_clear")=true,pybind11::arg("build_mode")=0)')
text=text.replace('.def("workspace_bytes",&FootprintGraphRasterizer::workspace_bytes)',
    '.def("snapshot",&FootprintGraphRasterizer::snapshot).def("workspace_bytes",&FootprintGraphRasterizer::workspace_bytes)')
path.write_text(text,encoding='utf-8');print('FOOTPRINT_GRAPH_GENERATED')
