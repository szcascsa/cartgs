"""恢复原排序对容量尾部的sentinel定义，保留失败源码和二进制。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_cooperative_sort_v2';shutil.copytree(HERE/'source_cooperative_sort',root);(HERE/'build_cooperative_sort_v2').mkdir()
    path=root/'cooperative_depth.cuh';text=path.read_text();anchor='  const int n=min(capacity,dynamic->P);'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
  // 背景敏感检查会读取整个容量前缀，尾部不能保留旧视角或未初始化ID。
  for(int index=n+blockIdx.x*blockDim.x+threadIdx.x;index<capacity;index+=gridDim.x*blockDim.x){
    output[index]=0xffffffff;ids[index]=index;if(inverse)inverse[index]=index;
  }
''');path.write_text(text)
    print('COOPERATIVE_PADDING_SOURCE_READY')


if __name__=='__main__':main()
