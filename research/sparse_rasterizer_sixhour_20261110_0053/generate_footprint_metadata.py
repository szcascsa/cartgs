"""把原预处理已算出的Tile矩形和opacity阈值保存到有界Workspace。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent;path=HERE/'source/forward_sparse.cu';source=path.read_text()
assert 'sixhour_preprocess_metadata' not in source
source=source.replace('template <int C, bool Keys = false, bool Strided = false>',
    'template <int C, bool Keys = false, bool Strided = false, bool Metadata = false>',1)
old='int v0=0,int v1=0,int p0=0,int p1=0,int c0=0) {'
new='int v0=0,int v1=0,int p0=0,int p1=0,int c0=0,uint4* footprint_rects=nullptr,float* footprint_limits=nullptr) {'
assert source.count(old)==1;source=source.replace(old,new,1)
old='  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);'
new=old+'''\n  if(Metadata) {
    footprint_rects[idx]=make_uint4(rect_min.x,rect_min.y,rect_max.x,rect_max.y);
    footprint_limits[idx]=logf(opacities[idx]/(1.0f/255.0f));
  }'''
assert source.count(old)==1;source=source.replace(old,new,1)
start=source.index('void sixhour_preprocess_strided(');stop=source.index('\n}',start)+2
wrapper=source[start:stop];signature=wrapper[:wrapper.index('{')]
signature=signature.replace('sixhour_preprocess_strided','sixhour_preprocess_metadata',1)
signature=signature.replace('int c0,bool keys)', 'int c0,uint4* footprint_rects,float* footprint_limits)')
launch=wrapper[wrapper.index('preprocessCUDA<NUM_CHAFFELS,true,true>'):wrapper.index(');')+2]
launch=launch.replace('preprocessCUDA<NUM_CHAFFELS,true,true>','preprocessCUDA<NUM_CHAFFELS,true,true,true>')
launch=launch.replace('depth_ids,v0,v1,p0,p1,c0);','depth_ids,v0,v1,p0,p1,c0,footprint_rects,footprint_limits);')
source+='\n\n'+signature+'{\n  '+launch+'\n}\n';path.write_text(source)
header=HERE/'source/fused_preprocess.h';header.write_text(header.read_text()+'\n'+signature.strip()+';\n')
print('FOOTPRINT_METADATA_GENERATED')
