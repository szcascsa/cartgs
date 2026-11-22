"""减少深度排序Graph节点，严格稳定排序；不按Ratio选择路径。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_cooperative_sort';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_cooperative_sort').mkdir()
    shutil.copyfile(HERE/'cooperative_depth.cuh',root/'cooperative_depth.cuh')
    path=root/'graph_main.cu';text=path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "cooperative_depth.cuh"')
    text=text.replace('build_mode<=11','build_mode<=17')
    text=text.replace('build_mode_==10||build_mode_==11','build_mode_==10||build_mode_==11||build_mode_==17')
    text=text.replace('if(build_mode_==10)executePreprocessedTasks','if(build_mode_==10||build_mode_==17)executePreprocessedTasks')
    start=text.index('    if(capacity<=32)warpDepthSort32')
    end=text.index('\n    if(build_mode_',start)
    # 原排序包含归并循环和中间指针交换，直到后续成员生成分支为止。
    original=text[start:end]
    assert 'mergeDepthSort' in original
    replacement='''    if(build_mode_==17){
      int resident=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cooperativeDepthSort,128,0));
      int blocks=std::min((capacity+511)/512,resident*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
      TORCH_CHECK(blocks>0&&at::cuda::getCurrentDeviceProperties()->cooperativeLaunch,"Cooperative depth sort unsupported");
      void* args[]={&capacity,&key_in,&merge_in,&merge_out,&key_out,&sorted,&inverse,&inputs};
      C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cooperativeDepthSort),dim3(blocks),dim3(128),args,0,capture));
    }else{
'''+original+'\n    }'
    text=text[:start]+replacement+text[end:]
    text=text.replace('StudentEncodedSignRasterizer','StudentCooperativeSortRasterizer');path.write_text(text)
    assert 'build_mode<=17' in text
    print('COOPERATIVE_SORT_SOURCE_READY')


if __name__=='__main__':main()
