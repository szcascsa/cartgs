"""精确纯Rasterizer候选：有界CUB稳定32bit深度排序，替代多次块归并。"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def main():
    root = HERE / 'source_cub_depth'
    shutil.copytree(HERE / 'source_encoded_sign', root)
    (HERE / 'build_cub_depth').mkdir()
    helper = '''__global__ void cubDepthInverse(int n,const uint32_t* sorted,uint32_t* inverse){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)inverse[sorted[i]]=i;
}
'''
    (root / 'cub_depth.cuh').write_text(helper)
    path = root / 'graph_main.cu'
    text = path.read_text().replace('#include "alpha_parity.cuh"', '#include "alpha_parity.cuh"\n#include "cub_depth.cuh"')
    assert '#include "cub_depth.cuh"' in text
    text = text.replace('Tensor geometry,sort,bits,directory,lists,counts,tasks;',
        'Tensor geometry,sort,bits,directory,lists,counts,tasks,radix;')
    text = text.replace('{geometry,sort,bits,directory,lists,counts,tasks}',
        '{geometry,sort,bits,directory,lists,counts,tasks,radix}')
    text = text.replace('Tensor pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks;',
        'Tensor pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks,pool_radix;')
    text = text.replace('{pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks}',
        '{pool_geometry,pool_sort,pool_bits,pool_directory,pool_lists,pool_counts,pool_tasks,pool_radix}')
    text = text.replace('pool_tasks=Tensor();pool_metadata=Tensor();', 'pool_tasks=Tensor();pool_metadata=Tensor();pool_radix=Tensor();')
    anchor = '    pool_capacity=capacity;pool_W=W;pool_H=H;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''    size_t radix_bytes=0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,radix_bytes,static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),static_cast<const uint32_t*>(nullptr),static_cast<uint32_t*>(nullptr),capacity,0,32));
    TORCH_CHECK(radix_bytes<=128ull<<20,"Depth radix workspace exceeds 128 MiB");
    pool_radix=buffer(radix_bytes);
''' + anchor)
    anchor = '    plan->lists=pool_lists;plan->counts=pool_counts;plan->tasks=pool_tasks;'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + 'plan->radix=pool_radix;')
    start = text.index('    if(capacity<=32)warpDepthSort32')
    end = text.index('\n    if(build_mode_', start)
    assert 'mergeDepthSort' in text[start:end]
    text = text[:start] + '''    size_t radix_bytes=plan->radix.numel();
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(plan->radix.data_ptr(),radix_bytes,key_in,key_out,id_in,sorted,capacity,0,32,capture));
    if(inverse)cubDepthInverse<<<(capacity+255)/256,256,0,capture>>>(capacity,sorted,inverse);
''' + text[end:]
    path.write_text(text.replace('StudentEncodedSignRasterizer', 'StudentCubDepthRasterizer'))
    print('CUB_DEPTH_SOURCE_READY', flush=True)


if __name__ == '__main__':
    main()
