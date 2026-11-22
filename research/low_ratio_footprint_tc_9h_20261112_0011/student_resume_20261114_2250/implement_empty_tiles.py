"""实际空Tile在共享属性初始化和屏障前退出；保持背景合成算式。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_empty_tiles';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_empty_tiles').mkdir()
    for name in ['encoded_sign_pixels.cuh','encoded_sign_diagnosed.cuh']:
        path=root/name;text=path.read_text();anchor='  const int count=counts[tile];'
        assert text.count(anchor)==1
        text=text.replace(anchor,anchor+'''
  if(!count){
    #pragma unroll
    for(int p=0;p<2;++p)if(inside[p])for(int ch=0;ch<3;++ch)
      out[ch*H*W+(y+Step*p)*W+x]=fmaf(1.f,bg[ch],0.f);
    return;
  }
''')
        path.write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('StudentEncodedSignRasterizer','StudentEmptyTilesRasterizer');path.write_text(text)
    print('EMPTY_TILES_SOURCE_READY')


if __name__=='__main__':main()
