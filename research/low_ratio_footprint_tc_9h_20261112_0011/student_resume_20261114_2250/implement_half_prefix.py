"""近似Student执行候选：冻结输入线性投影以FP16缓存，Feature表和网络权重仍保留原值。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_half_prefix'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_half_prefix').mkdir()
    path = root / 'tensor_networks.cuh'
    text = path.read_text()
    anchor = 'template<int Mode,int Warps,bool Direct=false,bool Projected=false>\n__global__ void compensationTensor'
    assert text.count(anchor) == 1
    text = text.replace(anchor, 'template<int Mode,int Warps,bool Direct=false,bool Projected=false,bool HalfFixed=false>\n__global__ void compensationTensor')
    anchor = '    const float* fixed=static_cast<const float*>(packed.p[5]);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '    using Fixed=typename std::conditional<HalfFixed,__half,float>::type;\n    const Fixed* fixed=static_cast<const Fixed*>(packed.p[5]);')
    anchor = '      if(begin+p<n)x=__fadd_rn(fixed[ids[p]*32+c],dynamic[anchors[p]*32+c]);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''      if(begin+p<n){
        float frozen;if constexpr(HalfFixed)frozen=__half2float(fixed[ids[p]*32+c]);else frozen=fixed[ids[p]*32+c];
        x=__fadd_rn(frozen,dynamic[anchors[p]*32+c]);
      }''')
    path.write_text(text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text().replace('compensation_mode!=9', 'compensation_mode!=9&&compensation_mode!=14')
    text = text.replace('if(compensation_mode==9){', 'if(compensation_mode==9||compensation_mode==14){')
    anchor = next(line for line in text.splitlines() if 'compensationTensor<Mode,Warps,Direct,Projected><<<' in line)
    text = text.replace(anchor,
        anchor.replace('compensationTensor<Mode,Warps,Direct,Projected>', 'if(compensation_mode==14)compensationTensor<Mode,Warps,Direct,Projected,true>') + '\n' + anchor.replace('compensationTensor<', 'else compensationTensor<'))
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('compensator_mode<=9', 'compensator_mode<=14')
    anchor = '      tensor_weights_[7]=torch::empty({state_[0].size(0),32},state_[0].options());'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n      if(tensor_compensator_==14)tensor_weights_[5]=tensor_weights_[5].to(at::kHalf);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-half-prefix.shared.v1').replace('StudentJointFinalRasterizer', 'StudentHalfPrefixRasterizer'))
    print('HALF_PREFIX_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
