"""精确Tensor候选：每帧按Anchor计算Selector动态首层，逐Gaussian只加冻结静态投影。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_selector_anchor'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_selector_anchor').mkdir()
    helper = '''template<int Width> __device__ __forceinline__ float selectorAnchorValue(int64_t row,int64_t n,int col,const float* view,float ratio,bool local,bool real,const bool* visible){
  const int j=col+10;if(row>=n||j>=Width||(visible&&!visible[row]))return 0.f;
  if(j==10)return ratio;if(j==18)return local?1.f:0.f;
  return local&&real?view[row*7+j-11]:0.f;
}
template<int Width> __global__ void selectorAnchorProjection(int64_t n,const float* view,const DeploymentFrame* frame,bool local,bool real,const void* weight,float* output,const bool* visible){
  if(!frame->k)return;
  const int lane=threadIdx.x%32,group=lane/4,t=lane%4;
  const int64_t begin=(int64_t(blockIdx.x)*4+threadIdx.x/32)*16,r0=begin+group,r1=r0+8;
  if(begin>=n)return;
  const float ratio=frame->log_ratio;unsigned a[4];const int j=t*2;
  a[0]=halfPair(selectorAnchorValue<Width>(r0,n,j,view,ratio,local,real),selectorAnchorValue<Width>(r0,n,j+1,view,ratio,local,real));
  a[1]=halfPair(selectorAnchorValue<Width>(r1,n,j,view,ratio,local,real),selectorAnchorValue<Width>(r1,n,j+1,view,ratio,local,real));
  a[2]=halfPair(selectorAnchorValue<Width>(r0,n,j+8,view,ratio,local,real),selectorAnchorValue<Width>(r0,n,j+9,view,ratio,local,real));
  a[3]=halfPair(selectorAnchorValue<Width>(r1,n,j+8,view,ratio,local,real),selectorAnchorValue<Width>(r1,n,j+9,view,ratio,local,real));
  #pragma unroll
  for(int ob=0;ob<4;++ob){
    float c[4]{};registerProduct<32,16>(a,static_cast<const __half*>(weight),0,ob,c);
    const int col=ob*8+t*2;
    if(r0<n){output[r0*32+col]=c[0];output[r0*32+col+1]=c[1];}
    if(r1<n){output[r1*32+col]=c[2];output[r1*32+col+1]=c[3];}
  }
}
'''
    helper = helper.replace('view,ratio,local,real)', 'view,ratio,local,real,visible)')
    text = (root / 'tensor_register.cuh').read_text()
    text = text[text.index('template<int Warps,int Width> __device__ __forceinline__ void selectorRegisterCore'):]
    text = text.replace('selectorRegisterCore', 'selectorProjected').replace('__device__ __forceinline__ void', '__global__ void')
    text = text.replace('const void* dynamic_weight', 'const float* projected')
    start = text.index('  unsigned a[4];int j=t*2;')
    end = text.index('  float first[4][4]{};', start)
    text = text[:start] + '  unsigned a[4];\n' + text[end:]
    anchor = '    registerProduct<32,16>(a,static_cast<const __half*>(dynamic_weight),0,ob,first[ob]);'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '')
    for number, row, extra in [(0, 'r0', ''), (1, 'r0', '+1'), (2, 'r1', ''), (3, 'r1', '+1')]:
        anchor = f'first[ob][{number}]=({row}<n?prefix[id{0 if row=="r0" else 1}*32+c{extra}]:0.f)+first[ob][{number}];'
        assert text.count(anchor) == 1
        replacement = f'first[ob][{number}]=({row}<n?prefix[id{0 if row=="r0" else 1}*32+c{extra}]:0.f)+({row}<n?projected[owners[id{0 if row=="r0" else 1}]*32+c{extra}]:0.f);'
        text = text.replace(anchor, replacement)
    (root / 'selector_anchor.cuh').write_text(helper + text)
    path = root / 'deployment_runtime_kernels.cu'
    text = path.read_text().replace('#include "tensor_networks.cuh"', '#include "tensor_networks.cuh"\n#include "selector_anchor.cuh"')
    text = text.replace('compensation_mode!=9', 'compensation_mode!=9&&compensation_mode!=13')
    text = text.replace('if(compensation_mode==9){', 'if(compensation_mode==9||compensation_mode==13){')
    text = text.replace('tensor_weights.size()==8', 'tensor_weights.size()>=8')
    anchor = '  const auto rows=buffers[1].data_ptr<int64_t>();'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''  if(compensation_mode==13){
    TORCH_CHECK(selector_mode==2&&tensor_weights.size()==9,"Anchor projection requires existing FP16 Selector and workspace");
    if(selector[0].size(0)==19)selectorAnchorProjection<19><<<(anchors+63)/64,128,0,stream>>>(anchors,buffers[2].data_ptr<float>(),frame,local,schema==2||schema==3,tensor_weights[4].data_ptr(),tensor_weights[8].data_ptr<float>(),visible.defined()?visible.data_ptr<bool>():nullptr);
    else selectorAnchorProjection<11><<<(anchors+63)/64,128,0,stream>>>(anchors,buffers[2].data_ptr<float>(),frame,local,false,tensor_weights[4].data_ptr(),tensor_weights[8].data_ptr<float>(),visible.defined()?visible.data_ptr<bool>():nullptr);
  }
''' + anchor)
    anchor = '      if(selector[0].size(0)==19)selectorTensor<Mode,Warps,19>'
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    assert len(matches) == 1
    index = matches[0]
    original = '\n'.join(lines[index:index+2])
    calls = []
    for width in [19, 11]:
        calls.append(f'        {"if" if width==19 else "else if"}(selector[0].size(0)=={width})selectorProjected<Warps,{width}><<<(n+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(n,rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.data_ptr<float>(),tensor_weights[0].data_ptr(),tensor_weights[8].data_ptr<float>());')
    replacement = '      if(compensation_mode==13){\n' + '\n'.join(calls) + '\n      }else{\n' + original + '\n      }'
    text = text.replace(original, replacement)
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('compensator_mode<=9', 'compensator_mode<=13')
    text = text.replace('tensor_weights_.resize(projection?8:5);', 'tensor_weights_.resize(projection?(tensor_compensator_==13?9:8):5);')
    anchor = '      tensor_weights_[7]=torch::empty({state_[0].size(0),32},state_[0].options());'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '\n      if(tensor_compensator_==13)tensor_weights_[8]=torch::empty_like(tensor_weights_[7]);')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-selector-anchor.shared.v1').replace('StudentJointFinalRasterizer', 'StudentSelectorAnchorRasterizer'))
    print('SELECTOR_ANCHOR_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
