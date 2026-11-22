"""只修改Student kernel线程与任务组织，原MMA、数学表达式及发布语义复用。"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source'
    path = root / 'tensor_networks.cuh'
    text = path.read_text()
    start = text.index('template<int Mode,int Warps,bool Direct=false,bool Projected=false>')
    original = text[start:]
    assert original.rstrip().endswith('}')
    candidate = original.replace(
        'template<int Mode,int Warps,bool Direct=false,bool Projected=false>',
        'template<int Mode,int Warps,bool Direct=false,bool Projected=false>')
    candidate = candidate.replace('compensationTensor(', 'compensationStudentScheduled(')
    candidate = candidate.replace('begin=blockIdx.x*Points;', 'first_begin=blockIdx.x*Points;')
    candidate = candidate.replace('if(begin>=n)return;', 'if(first_begin>=n)return;')
    anchor = '  if(tid<Points){ids[tid]'
    assert anchor in candidate
    candidate = candidate.replace(anchor,
        '  for(int64_t begin=first_begin;begin<n;begin+=int64_t(gridDim.x)*Points){\n' + anchor)
    anchor = '''  const int p=tid/8;
  if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],values[p],master,softplus,output,status,tid%8,8,request);'''
    assert anchor in candidate
    candidate = candidate.replace(anchor, '''  for(int p=tid/8;p<Points;p+=blockDim.x/8)
    if(begin+p<n)publishPoint<Direct,false>(begin+p,n,ids[p],values[p],master,softplus,output,status,tid%8,8,request);''')
    end = candidate.rfind('}')
    candidate = candidate[:end] + '  __syncthreads();\n  }\n' + candidate[end:]
    (root / 'student_scheduled.cuh').write_text(candidate)
    path.write_text(text + '\n#include "student_scheduled.cuh"\n')
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('compensator_mode<=2', 'compensator_mode<=6')
    text = text.replace('mycartgs.pending-raster.shared.pose.v3', 'mycartgs.student-schedule.shared.v1')
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('PosePendingNineRasterizer', 'StudentScheduleRasterizer')
    text = text.replace('mycartgs.pending-raster.shared.pose.v3', 'mycartgs.student-schedule.shared.v1')
    path.write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text().replace('#include <torch/extension.h>',
        '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>')
    anchor = 'int mode=(which==0||which==4)?selector_mode:compensation_mode;if(!mode)continue;'
    assert anchor in text
    text = text.replace(anchor, anchor + '\n    if(mode>=3)mode=2;')
    anchor = '''            compensationTensor<Mode,Warps,Direct,Projected><<<(capacity+Warps*16-1)/(Warps*16),Warps*128,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);'''
    assert anchor in text
    replacement = '''            if(compensation_mode>=3){
              const bool persistent=compensation_mode==4||compensation_mode==6;
              const int factor=compensation_mode==5?2:compensation_mode==6?4:1;
              const int64_t blocks=(capacity+Warps*16-1)/(Warps*16);
              const int64_t grid=persistent?std::min<int64_t>(blocks,4*at::cuda::getCurrentDeviceProperties()->multiProcessorCount):blocks;
              compensationStudentScheduled<Mode,Warps,Direct,Projected><<<grid,Warps*32*factor,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
            }else{
''' + anchor + '\n            }'
    text = text.replace(anchor, replacement)
    text = text.replace('dispatchTensor(compensation_mode,tensor_comp_warps,launch);',
                        'dispatchTensor(compensation_mode>=3?2:compensation_mode,tensor_comp_warps,launch);')
    path.write_text(text)
    print('STUDENT_SCHEDULE_SOURCE_READY', '32/64 threads per warp group; bounded persistent blocks')


if __name__ == '__main__':
    main()
