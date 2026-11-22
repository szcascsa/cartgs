"""精确选择的小候选集收尾：避免剩余几轮全Grid同步，稳定ID语义不变。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_radix'
    shutil.copytree(HERE / 'source_preparation', root)
    (HERE / 'build_radix').mkdir()
    path = root / 'deployment_radix.cu'
    text = path.read_text().replace('#include <cub/block/block_scan.cuh>',
        '#include <cub/block/block_scan.cuh>\n#include <cub/block/block_radix_sort.cuh>')
    text = text.replace('__global__ void shrinkingRadixFused(', 'template<bool SmallFinish> __global__ void shrinkingRadixFused(')
    start = text.index('template<bool SmallFinish>')
    end = text.index('__global__ void countThreshold', start)
    part = text[start:end]
    anchor = '      auto swap=input;input=output;output=swap;'
    assert part.count(anchor) == 1
    part = part.replace(anchor, anchor + '''
      if constexpr(SmallFinish){
        if(state[0]<=256){
          if(!blockIdx.x){
            using FinishSort=cub::BlockRadixSort<unsigned,threads,1>;
            using FinishReduce=cub::BlockReduce<int,threads>;
            __shared__ union {typename FinishSort::TempStorage sort;typename FinishReduce::TempStorage reduce;} finish;
            __shared__ unsigned finish_keys[threads];
            unsigned item[1]={tid<state[0]?input[tid]:0u};
            FinishSort(finish.sort).SortDescending(item);
            finish_keys[tid]=item[0];__syncthreads();
            const unsigned threshold=finish_keys[state[1]-1];
            const int greater=FinishReduce(finish.reduce).Sum(int(item[0]>threshold));
            if(!tid){state[2]=threshold;state[1]-=greater;}
          }
          grid.sync();
          break;
        }
      }
''')
    text = text[:start] + part + text[end:]
    text = text.replace('bool saturated_fast,bool fused=false) {', 'bool saturated_fast,bool fused=false,bool small_finish=false) {')
    text = text.replace('reinterpret_cast<void*>(shrinkingRadixFused),blocks,threads,fused_args,0,stream)',
        '(small_finish?reinterpret_cast<void*>(shrinkingRadixFused<true>):reinterpret_cast<void*>(shrinkingRadixFused<false>)),blocks,threads,fused_args,0,stream)')
    text = text.replace('bool saturated_fast,bool fused) {', 'bool saturated_fast,bool fused,bool small_finish) {')
    text = text.replace('),saturated_fast,fused);', '),saturated_fast,fused,small_finish);')
    text += '''
torch::Tensor deployment_radix_small(const torch::Tensor& scores,const torch::Tensor& rows,
    const torch::Tensor& eligible,int64_t k,const std::vector<torch::Tensor>& scratch,const torch::Tensor& status){
  auto output=torch::empty({k},rows.options());if(!k)return output;
  launchRadix(scores,rows,eligible,k,scratch,status,output,nullptr,false,true,true);return output;
}
'''
    path.write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text()
    text = text.replace('const std::vector<torch::Tensor>&,bool,bool);', 'const std::vector<torch::Tensor>&,bool,bool,bool);')
    text = text.replace('int64_t tensor_warps,int64_t tensor_comp_warps) {', 'int64_t tensor_warps,int64_t tensor_comp_warps,bool small_finish) {')
    text = text.replace('saturated_selection,fused_selection && n>=16384);', 'saturated_selection,fused_selection && n>=16384,small_finish);')
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    anchor = 'const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,int64_t,int64_t,int64_t,int64_t);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor[:-2] + ',bool);')
    text = text.replace('class TensorDeploymentPipeline {', 'class TensorDeploymentPipeline {\n  bool small_radix_finish_=false;')
    text = text.replace('tensor_compensator_,tensor_warps_,tensor_comp_warps_);', 'tensor_compensator_,tensor_warps_,tensor_comp_warps_,small_radix_finish_);')
    text = text.replace('  torch::Tensor debug_control() const',
        '  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}\n  torch::Tensor debug_control() const')
    text = text.replace('.def("_control",&TensorDeploymentPipeline::debug_control)',
        '.def("_control",&TensorDeploymentPipeline::debug_control)\n    .def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish)')
    text = text.replace('mycartgs.student-preparation.shared.v1', 'mycartgs.student-radix.shared.v1')
    path.write_text(text)
    path = root / 'deployment_runtime_bindings.cpp'
    text = path.read_text()
    text = text.replace('void register_pending_raster(pybind11::module_&);', '''torch::Tensor deployment_radix_small(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const std::vector<torch::Tensor>&,const torch::Tensor&);
void register_pending_raster(pybind11::module_&);''')
    text = text.replace('  module.def("_radix_fused", &deployment_radix_fused);',
        '  module.def("_radix_fused", &deployment_radix_fused);\n  module.def("_radix_small", &deployment_radix_small);')
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('StudentPreparationRasterizer', 'StudentRadixRasterizer')
    text = text.replace('mycartgs.student-preparation.shared.v1', 'mycartgs.student-radix.shared.v1')
    path.write_text(text)
    print('STUDENT_RADIX_SOURCE_READY')


if __name__ == '__main__':
    main()
