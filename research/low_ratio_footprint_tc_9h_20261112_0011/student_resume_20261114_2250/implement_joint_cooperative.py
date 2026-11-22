"""在动态GPU K的完整推理上单独测试协作排序，排除纯渲染已知K的调度差异。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_joint_cooperative';shutil.copytree(HERE/'source_joint_final',root);(HERE/'build_joint_cooperative').mkdir()
    shutil.copyfile(HERE/'source_cooperative_sort_v2/cooperative_depth.cuh',root/'raster/cooperative_depth.cuh')
    path=root/'raster/graph_main.cu';text=path.read_text().replace('#include "pending_dynamic.cuh"',
        '#include "pending_dynamic.cuh"\n#include "cooperative_depth.cuh"')
    text=text.replace('build_mode<=15','build_mode<=17')
    text=text.replace('build_mode_==14||build_mode_==15','build_mode_==14||build_mode_==15||build_mode_==17')
    text=text.replace('build_mode_==13||build_mode_==15','build_mode_==13||build_mode_==15||build_mode_==17')
    start=text.index('    if(capacity<=32)warpDepthSort32');end=text.index('\n    if(build_mode_',start)
    original=text[start:end];assert 'mergeDepthSort' in original
    replacement='''    if(build_mode_==17){
      int resident=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,cooperativeDepthSort,128,0));
      int blocks=std::min((capacity+511)/512,resident*at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
      TORCH_CHECK(blocks>0&&at::cuda::getCurrentDeviceProperties()->cooperativeLaunch,"Cooperative depth sort unsupported");
      void* args[]={&capacity,&key_in,&merge_in,&merge_out,&key_out,&sorted,&inverse,&inputs};
      C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(cooperativeDepthSort),dim3(blocks),dim3(128),args,0,capture));
    }else{
'''+original+'\n    }'
    text=text[:start]+replacement+text[end:]
    text=text.replace('StudentJointFinalRasterizer','StudentJointCooperativeRasterizer').replace('mycartgs.student-joint-final.shared.v1','mycartgs.student-joint-cooperative.shared.v1');path.write_text(text)
    path=root/'deployment_prepare.cpp';text=path.read_text().replace('mycartgs.student-joint-final.shared.v1','mycartgs.student-joint-cooperative.shared.v1');path.write_text(text)
    print('JOINT_COOPERATIVE_SOURCE_READY')


if __name__=='__main__':main()
