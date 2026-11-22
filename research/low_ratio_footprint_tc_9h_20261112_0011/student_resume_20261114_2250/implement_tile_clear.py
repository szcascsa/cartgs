"""精确调度候选：每CTA一个Tile清零实际K对应位表，避免逐字动态整数除法。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_tile_clear'
    shutil.copytree(HERE / 'source_joint_final', root)
    (HERE / 'build_tile_clear').mkdir()
    header = '''template<int Threads> __global__ void clearTilePendingBits(uint32_t* bits,int words,int tiles,const GraphInputs* input){
  const int active=(input->P+31)/32,tile=blockIdx.x;
  if(!tile&&!threadIdx.x){bits[int64_t(tiles)*words]=0;bits[int64_t(tiles)*words+1]=0;}
  for(int word=threadIdx.x;word<active;word+=Threads)bits[int64_t(tile)*words+word]=0;
}
'''
    (root / 'raster/tile_clear.cuh').write_text(header)
    path = root / 'raster/graph_main.cu'
    text = path.read_text().replace('#include "pending_dynamic.cuh"', '#include "pending_dynamic.cuh"\n#include "tile_clear.cuh"')
    text = text.replace('build_mode<=15', 'build_mode<=19')
    text = text.replace('build_mode_==13||build_mode_==15', 'build_mode_==13||build_mode_==15||build_mode_>=16')
    anchor = '    if(build_mode_==14||build_mode_==15)clearActivePendingBits'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    if(build_mode_==16)clearTilePendingBits<64><<<tiles,64,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==17)clearTilePendingBits<128><<<tiles,128,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==18)clearTilePendingBits<256><<<tiles,256,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==19)clearTilePendingBits<32><<<tiles,32,0,capture>>>(bits,words,tiles,inputs);
    else if(build_mode_==14||build_mode_==15)clearActivePendingBits''')
    path.write_text(text)
    for name in ['deployment_prepare.cpp', 'raster/graph_main.cu']:
        path = root / name
        path.write_text(path.read_text().replace('mycartgs.student-joint-final.shared.v1',
            'mycartgs.student-tile-clear.shared.v1').replace('StudentJointFinalRasterizer', 'StudentTileClearRasterizer'))
    print('TILE_CLEAR_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
