"""精确准备候选：8/16/32 lane一Anchor，保持角点归约和每列投影顺序。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_grouped_view'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_grouped_view').mkdir()
    text = (root / 'student_view_project.cuh').read_text()
    text = text.replace('__global__ void studentViewProject', 'template<int Lanes> __global__ void studentGroupedViewProject')
    text = text.replace('const int lane=threadIdx.x%32;',
        'const int lane=threadIdx.x%Lanes;\n  const unsigned group_mask=(0xffffffffu>>(32-Lanes))<<((threadIdx.x%32)/Lanes*Lanes);')
    text = text.replace('(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32',
        '(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)/Lanes')
    text = text.replace('if(!a){projected[lane]=0.f;if(lane<7)out[lane]=0.f;}',
        'if(!a){for(int col=lane;col<32;col+=Lanes)projected[col]=0.f;if(lane<7)out[lane]=0.f;}')
    for name in ['lo0', 'hi0', 'lo1', 'hi1', 'mind', 'maxd']:
        text = text.replace(f'__shfl_xor_sync(0xffffffff,{name},delta)',
                            f'__shfl_xor_sync(group_mask,{name},delta,Lanes)')
    anchor = '  float sum=0.f;'
    assert text.count(anchor) == 1
    text = text[:text.index(anchor)] + '''  float encoded[8];
  #pragma unroll
  for(int j=0;j<8;++j)encoded[j]=j==7?1.f:(real_view?__shfl_sync(group_mask,value,j,Lanes):0.f);
  #pragma unroll
  for(int part=0;part<32/Lanes;++part){
    const int col=lane+part*Lanes;float sum=0.f;
    for(int j=0;j<levels;++j){const float weight=frame->weights[j+1];if(weight<0.f)break;
      sum=__fadd_rn(sum,__fmul_rn(weight,residual[(int64_t(j)*n+a)*32+col]));}
    for(int j=0;j<3;++j)sum=__fmaf_rn(frame->ratio_code[j],weights.p[4][j*32+col],sum);
    #pragma unroll
    for(int j=0;j<8;++j)sum=__fmaf_rn(encoded[j],view_weight[j*32+col],sum);
    projected[a*32+col]=sum;
  }
}
'''
    (root / 'student_grouped_view.cuh').write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text().replace('#include "student_view_project.cuh"',
        '#include "student_view_project.cuh"\n#include "student_grouped_view.cuh"')
    assert text.count('compensation_mode!=9') == 2
    text = text.replace('compensation_mode!=9', 'compensation_mode<9')
    text = text.replace('if(compensation_mode==9){', 'if(compensation_mode>=9){')
    line = next(line for line in text.splitlines() if 'studentViewProject<<<' in line)
    added = []
    for mode, lanes in [(10, 8), (11, 16), (12, 32)]:
        added.append(line.replace('studentViewProject<<<(anchors*32+127)',
            f'if(compensation_mode=={mode})studentGroupedViewProject<{lanes}><<<(anchors*{lanes}+127)'))
    text = text.replace(line, line.replace('    studentViewProject<<<', '    if(compensation_mode==9)studentViewProject<<<') + '\n' + '\n'.join(added))
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text()
    assert text.count('compensator_mode<=9') == 1
    path.write_text(text.replace('compensator_mode<=9', 'compensator_mode<=12'))
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-grouped-view.shared.v1').replace('StudentJointFinalRasterizer', 'StudentGroupedViewRasterizer'))
    print('GROUPED_VIEW_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
