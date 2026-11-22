"""保留预处理数学表达式，生成只多写整数排序元数据的独立入口。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/'source/forward_sparse.cu'
source=path.read_text()
assert 'sixhour_preprocess_keys' not in source
begin=source.index('template <int C>\n__global__ void preprocessCUDA')
end=source.index('// Main rasterization method.',begin)
kernel=source[begin:end]
kernel=kernel.replace('template <int C>','template <int C, bool Keys = false>',1)
assert kernel.count('bool prefiltered) {')==1
kernel=kernel.replace('bool prefiltered) {','bool prefiltered,uint32_t* depth_keys=nullptr,uint32_t* depth_ids=nullptr) {',1)
kernel=kernel.replace('if (idx >= P) return;','if (idx >= P) return;\n  if(Keys){depth_keys[idx]=0xffffffff;depth_ids[idx]=idx;}',1)
kernel=kernel.replace('depths[idx] = p_view.z;','depths[idx] = p_view.z;\n  if(Keys)depth_keys[idx]=__float_as_uint(p_view.z);',1)
source=source[:begin]+kernel+source[end:]
start=source.index('void FORWARD::preprocess(')
stop=source.index('\n}',start)+2
wrapper=source[start:stop].replace('void FORWARD::preprocess(','void sixhour_preprocess_keys(',1)
wrapper=wrapper.replace('bool prefiltered) {','bool prefiltered,uint32_t* depth_keys,uint32_t* depth_ids) {',1)
wrapper=wrapper.replace('preprocessCUDA<NUM_CHAFFELS>','preprocessCUDA<NUM_CHAFFELS,true>',1)
assert wrapper.count('grid, tiles_touched, prefiltered);')==1
wrapper=wrapper.replace('grid, tiles_touched, prefiltered);','grid, tiles_touched, prefiltered,depth_keys,depth_ids);',1)
source+='\n\n'+wrapper+'\n'
path.write_text(source)
(HERE/'source/fused_preprocess.h').write_text('#pragma once\n'+wrapper[:wrapper.index('{')].strip()+';\n')
print('FUSED_PREPROCESS_GENERATED')
