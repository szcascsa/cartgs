"""计时外统计可筛除工作量，并逐像素查找任何错误筛除。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
source=HERE/'source'
assert not (HERE/'source_first').exists()
shutil.copytree(source,HERE/'source_first')
shutil.copy2(HERE/'build/mycartgs_large_footprint_subtiles_v1.so',HERE/'reports/subtiles_first.so')
header=source/'subtile_pixels.cuh'
text=header.read_text()
text+="""
// 仅供计时外诊断，不影响render的任何执行节点。
__global__ void subtileDiagnostics(int tiles,int side,int W,int H,const uint32_t* lists,const uint32_t* counts,
    const float2* xy,const float4* conic,unsigned long long* counters,bool verify) {
  int index=blockIdx.x*blockDim.x+threadIdx.x,tile=index/256,j=index%256;
  if(tile>=tiles)return;
  const unsigned count=counts[tile];
  if(!j){atomicAdd(counters,static_cast<unsigned long long>(count));atomicAdd(counters+6,1ull);
    if(count>256)atomicAdd(counters+7,1ull);}
  if(count>256||unsigned(j)>=count)return;
  const int id=lists[tile*256+j],tx=tile%((W+15)/16),ty=tile/((W+15)/16);
  unsigned tests=0,keep=0,reject=0,wrong=0;
  for(int y=0;y<16;y+=side)for(int x=0;x<16;x+=side) {
    int xmin=tx*16+x,ymin=ty*16+y;
    ++tests;
    if(mayContributeRect(xy[id],conic[id],xmin,ymin,xmin+side-1,ymin+side-1)){++keep;continue;}
    ++reject;
    if(verify)for(int py=ymin;py<min(H,ymin+side);++py)for(int px=xmin;px<min(W,xmin+side);++px)
      if(alphaValue(xy[id],conic[id],make_float2(px,py))>=0.f)++wrong;
  }
  atomicAdd(counters+1,static_cast<unsigned long long>(tests));
  atomicAdd(counters+2,static_cast<unsigned long long>(keep));
  atomicAdd(counters+3,static_cast<unsigned long long>(reject));
  atomicAdd(counters+4,static_cast<unsigned long long>(wrong));
  atomicAdd(counters+5,static_cast<unsigned long long>(keep*side*side));
}
"""
header.write_text(text)
main=source/'graph_main.cu'
text=main.read_text()
method="""
  Tensor subtile_statistics(int side,bool verify)const {
    TORCH_CHECK(side==4||side==8,"Subtile size must be 4 or 8");
    TORCH_CHECK(!plans.empty(),"Render before diagnostics");
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==owner,"One renderer per stream");
    auto found=std::max_element(plans.begin(),plans.end(),[](const auto& a,const auto& b){return a->last_used<b->last_used;});
    const auto& p=**found;const int n=p.capacity,tiles=((p.W+15)/16)*((p.H+15)/16);
    char* geom=reinterpret_cast<char*>(p.geometry.data_ptr());
    float *depths,*covariance,*rgb;bool* clamped;int* radii;float2* xy;float4* conic;uint32_t* touched;
    using CudaRasterizer::obtain;
    obtain(geom,depths,n,128);obtain(geom,clamped,n*3,128);obtain(geom,radii,n,128);
    obtain(geom,xy,n,128);obtain(geom,covariance,n*6,128);obtain(geom,conic,n,128);
    obtain(geom,rgb,n*3,128);obtain(geom,touched,n,128);
    auto result=torch::zeros({8},p.counts.options().dtype(at::kLong));
    subtileDiagnostics<<<tiles,256,0,owner>>>(tiles,side,p.W,p.H,
        reinterpret_cast<const uint32_t*>(p.lists.data_ptr()),reinterpret_cast<const uint32_t*>(p.counts.data_ptr()),
        xy,conic,reinterpret_cast<unsigned long long*>(result.data_ptr()),verify);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
  }
"""
text=text.replace('  std::vector<int64_t> stats()const',method+'\n  std::vector<int64_t> stats()const')
text=text.replace('.def("stats",&SubtileRasterizer::stats);',
                  '.def("stats",&SubtileRasterizer::stats).def("subtile_statistics",&SubtileRasterizer::subtile_statistics);')
assert 'def("subtile_statistics"' in text
main.write_text(text)
print('DIAGNOSTICS_READY',flush=True)
