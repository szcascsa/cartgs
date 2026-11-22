"""输入发布与位表清零合一；当前帧参数传入，无Host计数读取或旧帧缓存。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_update_clear';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_update_clear').mkdir()
    path=root/'graph_main.cu';text=path.read_text()
    anchor='#include "encoded_sign_diagnosed.cuh"'
    helper='''
__global__ void updateAndClearGraphInputs(GraphInputs value,GraphInputs* destination,uint32_t* bits,int64_t count){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(!i)*destination=value;
  for(;i<count;i+=int64_t(gridDim.x)*blockDim.x)bits[i]=0;
}
'''
    assert anchor in text;text=text.replace(anchor,anchor+'\n'+helper)
    text=text.replace('build_mode<=11','build_mode<=16')
    text=text.replace('build_mode_==10||build_mode_==11','build_mode_==10||build_mode_==11||build_mode_==16')
    text=text.replace('if(fused_tasks)clearGraphBits','if(fused_tasks&&build_mode_!=16)clearGraphBits')
    text=text.replace('if(build_mode_==10)executePreprocessedTasks','if(build_mode_==10||build_mode_==16)executePreprocessedTasks')
    old='} else updateGraphInputs<<<1,1,0,stream>>>(inputs,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));'
    assert text.count(old)==1
    new='''} else if(build_mode_==16){
      const int64_t count=int64_t((W+15)/16)*((H+15)/16)*((capacity+31)/32)+2;
      const int blocks=std::min<int64_t>((count+255)/256,at::cuda::getCurrentDeviceProperties()->multiProcessorCount*8);
      updateAndClearGraphInputs<<<blocks,256,0,stream>>>(inputs,reinterpret_cast<GraphInputs*>(parameters.data_ptr()),reinterpret_cast<uint32_t*>(selected->bits.data_ptr()),count);
    } else updateGraphInputs<<<1,1,0,stream>>>(inputs,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));'''
    text=text.replace(old,new)
    anchor='    TORCH_CHECK(pool_capacity>=capacity&&pool_W==W&&pool_H==H,"Graph pool must be reserved before capture");'
    text=text.replace(anchor,anchor+'\n    TORCH_CHECK(build_mode_!=16||!input_in_graph,"Fused input/clear uses current host input launch");')
    text=text.replace('StudentEncodedSignRasterizer','StudentUpdateClearRasterizer');path.write_text(text)
    assert 'build_mode<=16' in text
    print('UPDATE_CLEAR_SOURCE_READY')


if __name__=='__main__':main()
