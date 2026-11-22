"""联合候选增量：实际GPU K限制排序和位表工作，独立build_mode=12。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def function(text,signature):
    start=text.index(signature);brace=text.index('{',start);depth=1;end=brace+1
    while depth:
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    return text[start:end]


def main():
    root=HERE/'joint_source/raster';text=(root/'graph_kernels.cuh').read_text()
    pieces=[]
    for original,new in [('blockDepthSort512','blockDepthSort512Pending'),('mergeDepthSort','mergeDepthSortPending')]:
        f=function(text,'__global__ void '+original+'(').replace(original,new)
        f=f.replace('uint32_t* inverse=nullptr) {','uint32_t* inverse,const GraphInputs* dynamic) {')
        if original=='blockDepthSort512':
            body='''
  n=min(n,dynamic->P);
  if(!n){if(!blockIdx.x&&!threadIdx.x)output[0]=0xffffffff;return;}
  if(blockIdx.x*512>=unsigned(n))return;
  final=n<=512;
'''
        else:body='''
  n=min(n,dynamic->P);if(width>=n)return;final=width*2>=n;
'''
        pos=f.index('{')+1;f=f[:pos]+body+f[pos:];pieces.append(f)
    f=function(text,'__device__ bool copySparseBackground(').replace('copySparseBackground','copyPendingBackground')
    f=f.replace('if(keys[0]==0xffffffff)return true;','n=min(n,dynamic->P);if(keys[0]==0xffffffff)return true;')
    f=f.replace('const int visible_words=min(words,8);','const int visible_words=min((n+31)/32,8);')
    pieces.append(f)
    f=function(text,'__global__ void compactTileBits(').replace('compactTileBits','compactPendingBits')
    f=f.replace('int lane=threadIdx.x&31,','const int active_words=(min(n,dynamic->P)+31)/32;\n  int lane=threadIdx.x&31,')
    f=f.replace('w<words?bits','w<active_words?bits').replace('copySparseBackground','copyPendingBackground')
    pieces.append(f)
    pieces.append('''__global__ void clearPendingBits(uint32_t* bits,int words,int tiles,const GraphInputs* dynamic){
  const int i=blockIdx.x*blockDim.x+threadIdx.x,active_words=(dynamic->P+31)/32;
  if(!i){bits[int64_t(tiles)*words]=0;bits[int64_t(tiles)*words+1]=0;}
  if(i<int64_t(tiles)*active_words)bits[int64_t(i/active_words)*words+i%active_words]=0;
}''')
    (root/'pending_dynamic.cuh').write_text('\n\n'.join(pieces))
    path=root/'graph_main.cu';s=path.read_text().replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "pending_dynamic.cuh"')
    s=s.replace('build_mode<=11','build_mode<=12').replace('build_mode_==10||build_mode_==11','build_mode_==10||build_mode_==11||build_mode_==12')
    s=s.replace('pool_bits=buffer(bit_bytes);','pool_bits=buffer(bit_bytes);pool_bits.zero_();')
    anchor='''    if(fused_tasks)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);'''
    assert anchor in s;s=s.replace(anchor,'''    if(build_mode_==12)clearPendingBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,words,tiles,inputs);
    else if(fused_tasks)clearGraphBits<<<(bit_bytes/4+255)/256,256,0,capture>>>(bits,bit_bytes/4);''')
    anchor='''    else blockDepthSort512<<<(capacity+511)/512,128,0,capture>>>(capacity,key_in,merge_in,key_out,sorted,capacity<=512,inverse);'''
    assert anchor in s;s=s.replace(anchor,'''    else if(build_mode_==12)blockDepthSort512Pending<<<(capacity+511)/512,128,0,capture>>>(capacity,key_in,merge_in,key_out,sorted,capacity<=512,inverse,inputs);
'''+anchor)
    anchor='''      mergeDepthSort<<<(capacity+255)/256,256,0,capture>>>(capacity,width,merge_in,merge_out,key_out,sorted,width*2>=capacity,inverse);'''
    assert anchor in s;s=s.replace(anchor,'''      if(build_mode_==12)mergeDepthSortPending<<<(capacity+255)/256,256,0,capture>>>(capacity,width,merge_in,merge_out,key_out,sorted,width*2>=capacity,inverse,inputs);
      else '''+anchor.strip())
    s=s.replace('else if(build_mode_==10)executePreprocessedTasks<32>','else if(build_mode_==10||build_mode_==12)executePreprocessedTasks<32>')
    anchor=next(line for line in s.splitlines() if line.strip().startswith('compactTileBits<<<'))
    dynamic=anchor.replace('compactTileBits','compactPendingBits').replace('pixel_mode_>=18?inputs:nullptr','inputs')
    s=s.replace(anchor,'      if(build_mode_==12)'+dynamic.strip()+'\n      else '+anchor.strip())
    path.write_text(s)
    print('PENDING_ACTUAL_K_WORK_ADDED',12)


if __name__=='__main__':main()
