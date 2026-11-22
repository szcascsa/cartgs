"""当前GPU实际K控制清零与成员压缩，无ratio门限、不改变成员或排序。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_members'
    shutil.copytree(HERE / 'source_radix', root)
    (HERE / 'build_members').mkdir()
    path = root / 'raster/pending_dynamic.cuh'
    text = path.read_text()
    start = text.index('__global__ void compactPendingBits')
    original = text[start:text.index('__global__ void clearPendingBits', start)]
    active = original.replace('compactPendingBits(', 'compactActivePendingBits(')
    active = active.replace('for(int group=0;group<groups;++group)', 'for(int group=0;group<(active_words+31)/32;++group)')
    anchor = '  if(!lane)counts[tile]=carry;'
    active = active.replace(anchor, '''  // 未活跃directory保持原始全零语义，快照和溢出路径均可直接消费。
  for(int group=(active_words+31)/32+lane;group<groups;group+=32)directory[tile*groups+group]=0;
''' + anchor)
    text += '\n' + active + '''
__global__ void clearActivePendingBits(uint32_t* bits,int words,int tiles,const GraphInputs* dynamic){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int active_words=(dynamic->P+31)/32;
  if(!i){bits[int64_t(tiles)*words]=0;bits[int64_t(tiles)*words+1]=0;}
  const int64_t count=int64_t(tiles)*active_words;
  for(;i<count;i+=int64_t(gridDim.x)*blockDim.x)bits[(i/active_words)*words+i%active_words]=0;
}
'''
    path.write_text(text)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('build_mode<=12', 'build_mode<=15')
    text = text.replace('build_mode_==12', 'build_mode_>=12')
    text = text.replace('build_mode_==10||build_mode_==11||build_mode_>=12', 'build_mode_==10||build_mode_==11||build_mode_>=12')
    anchor = '    if(build_mode_>=12)clearPendingBits'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    if(build_mode_==14||build_mode_==15)clearActivePendingBits<<<std::min<int64_t>((bit_bytes/4+255)/256,workers*2),256,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_>=12)clearPendingBits''')
    anchor = '      if(build_mode_>=12)compactPendingBits'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''      if(build_mode_==13||build_mode_==15)compactActivePendingBits<<<(tiles+3)/4,128,0,capture>>>(bits,words,tiles,sorted,directory,groups,lists,counts,list_capacity,capacity,key_out,touched,inputs);
      else if(build_mode_>=12)compactPendingBits''')
    text = text.replace('StudentRadixRasterizer', 'StudentMembersRasterizer')
    text = text.replace('mycartgs.student-radix.shared.v1', 'mycartgs.student-members.shared.v1')
    path.write_text(text)
    path = root / 'deployment_prepare.cpp'
    text = path.read_text().replace('mycartgs.student-radix.shared.v1', 'mycartgs.student-members.shared.v1')
    path.write_text(text)
    print('STUDENT_ACTIVE_MEMBERS_SOURCE_READY')


if __name__ == '__main__':
    main()
