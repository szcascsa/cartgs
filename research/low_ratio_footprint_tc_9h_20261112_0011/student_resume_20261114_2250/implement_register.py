"""Student寄存器MMA独立源码/构建，模式7为普通网格，8为有界循环网格。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_register'
    shutil.copytree(HERE / 'source', root)
    (HERE / 'build_register').mkdir()
    shutil.copy2(HERE / 'student_register.cuh', root / 'student_register.cuh')
    path = root / 'tensor_networks.cuh'
    path.write_text(path.read_text() + '\n#include "student_register.cuh"\n')
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('compensator_mode<=6', 'compensator_mode<=8')
    text = text.replace('mycartgs.student-schedule.shared.v1', 'mycartgs.student-register.shared.v1')
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('StudentScheduleRasterizer', 'StudentRegisterRasterizer')
    text = text.replace('mycartgs.student-schedule.shared.v1', 'mycartgs.student-register.shared.v1')
    path.write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text()
    anchor = '    if(mode==1)packTensorWeights<1>'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    if(compensation_mode>=7&&(which==2||which==3)){
      packStudentRegisterWeights<<<(size+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(cw,which,reinterpret_cast<__half*>(result[which].data_ptr<at::Half>()));
      continue;
    }
''' + anchor)
    anchor = '            if(compensation_mode>=3){'
    assert anchor in text
    text = text.replace(anchor, '''            if(compensation_mode>=7){
              TORCH_CHECK(Projected,"Register Student requires frozen projected input");
              const int64_t blocks=(capacity+Warps*16-1)/(Warps*16);
              const int64_t grid=compensation_mode==8?std::min<int64_t>(blocks,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount):blocks;
              compensationStudentRegisters<Warps,Direct><<<grid,Warps*32,0,stream>>>(capacity,buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),cw,m,softplus,buffers[6].data_ptr<float>(),status,frame,destination,tw);
            }else if(compensation_mode>=3){''')
    path.write_text(text)
    print('STUDENT_REGISTER_SOURCE_READY')


if __name__ == '__main__':
    main()
