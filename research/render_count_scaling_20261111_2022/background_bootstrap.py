"""独立第二阶段，原冻结候选和所有原始测量保持不变。"""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
HERE=ROOT/'background_stage'
assert not HERE.exists()
HERE.mkdir()
shutil.copytree(ROOT/'source',HERE/'source')
for name in ('build','reports','logs'):(HERE/name).mkdir()
for name in ('launch.py','job_runner.py'):
    shutil.copy2(ROOT/name,HERE/name)
main=HERE/'source/graph_main.cu'
text=main.read_text().replace('CountScalingRasterizer','BackgroundScalingRasterizer')
text=text.replace('pixel_mode<=17','pixel_mode<=18')
line=next(x for x in text.splitlines() if 'if(pixel_mode_==14)renderScalingPixels<9>' in x)
text=text.replace(line,line+'\n'+line.replace('==14','==18'))
old='list_capacity,capacity,key_out,touched);'
assert old in text
text=text.replace(old,'list_capacity,capacity,key_out,touched,pixel_mode_==18?inputs:nullptr);')
main.write_text(text)
kernels=HERE/'source/graph_kernels.cuh'
text=kernels.read_text()
old='const uint32_t* keys,const uint32_t* touched,int lane) {'
new="""const uint32_t* keys,const uint32_t* touched,int lane,const GraphInputs* dynamic=nullptr) {
  if(dynamic) {
    // 只有非敏感背景可跳过全图空成员证明：原C=0,T=1的输出与直接copy完全相同。
    // 不接受负零、denormal、NaN/Inf；这些仍走原检查。
    bool ordinary=true;
    #pragma unroll
    for(int ch=0;ch<3;++ch){
      const uint32_t raw=__float_as_uint(dynamic->background[ch]),mag=raw&0x7fffffffu;
      ordinary=ordinary&&(raw==0||(mag>=0x00800000u&&mag<0x7f800000u));
    }
    if(ordinary)return false;
  }"""
assert text.count(old)==1
text=text.replace(old,new)
old='int n,const uint32_t* depth_keys,const uint32_t* touched) {'
assert text.count(old)==1
text=text.replace(old,'int n,const uint32_t* depth_keys,const uint32_t* touched,const GraphInputs* dynamic=nullptr) {')
text=text.replace('copySparseBackground(bits,words,tiles,n,sorted,depth_keys,touched,lane);',
                  'copySparseBackground(bits,words,tiles,n,sorted,depth_keys,touched,lane,dynamic);')
kernels.write_text(text)
print('BACKGROUND_STAGE_READY',flush=True)
