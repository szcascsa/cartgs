"""异常判断提升到Tile级，消除正常输入每成员fallback分支。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/DEV_CONTROLLER05.json').read_text())['state']=='complete'
    snapshot=HERE/'development_v4';snapshot.mkdir();shutil.copytree(HERE/'source',snapshot/'source')
    shutil.copy2(HERE/'build/mycartgs_footprint_ninehour_v1.so',snapshot/'mycartgs_footprint_ninehour_v1.so')
    shutil.copy2(HERE/'render_runtime.py',snapshot/'render_runtime.py')
    path=HERE/'source/folded_pixels.cuh';s=path.read_text()
    s=s.replace('Mode>=28&&Mode<=31?1:','Mode>=28&&Mode<=31||Mode>=37&&Mode<=39?1:')
    s=s.replace('Mode==30||Mode==31||Mode==32','Mode==30||Mode==31||Mode==32||Mode>=37')
    s=s.replace('Mode==29||Mode==31?.01f:','Mode==29||Mode==31||Mode==38?.01f:Mode==39?.02f:')
    s=s.replace('__shared__ unsigned valid[Mode>=28?256:1];','''__shared__ unsigned valid[Mode>=28&&Mode<=32?256:1];__shared__ unsigned unsafe_tile;
    if(Mode>=37){if(!tid)unsafe_tile=0;__syncthreads();}''')
    s=s.replace('if(Mode>=28){','if(Mode>=28&&Mode<=32){')
    anchor='      abc[j]=f.a;ef[j]=f.b;rgb[j]='
    replacement='''      if(Mode>=37){
        bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
        const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
        for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
        if(!safe)atomicOr(&unsafe_tile,1u);
      }
      abc[j]=f.a;ef[j]=f.b;rgb[j]='''
    assert anchor in s;s=s.replace(anchor,replacement)
    s=s.replace('if(Mode>=28&&!valid[j])','if(Mode>=28&&Mode<=32&&!valid[j])')
    anchor='''    __syncthreads();
    #pragma unroll 32'''
    replacement='''    __syncthreads();
    if(Mode>=37&&unsafe_tile){
      for(int j=0;j<count;++j){
        if((j&7)==0&&__all_sync(0xffffffff,done[0]&&done[1]))break;
        const int id=lists[tile*capacity+j];float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
        compositeScalingPair<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F);
      }
    }else{
    #pragma unroll 32'''
    assert anchor in s;s=s.replace(anchor,replacement)
    anchor='''  } else {
    for(int group'''
    assert anchor in s;s=s.replace(anchor,'  }\n'+anchor)
    s=s.replace('if(Mode>=28&&!safe)','if(Mode>=37||Mode>=28&&!safe)')
    path.write_text(s)
    path=HERE/'source/graph_main.cu';s=path.read_text().replace('pixel_mode<=54','pixel_mode<=58')
    anchor=next(line for line in s.splitlines() if 'if(pixel_mode_==54)renderTensorPixels<36>' in line)
    extra='\n'.join(f'      if(pixel_mode_=={18+m})renderFoldedPixels<{m}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);' for m in range(37,41))
    s=s.replace(anchor,anchor+'\n'+extra);path.write_text(s)
    path=HERE/'render_runtime.py';s=path.read_text();needle='\n\n\n@lru_cache'
    modes={37:'tile_safe_bitexp',38:'tile_safe_bitexp_stop',39:'tile_safe_bitexp_stop2',40:'tile_safe_fast'}
    s=s.replace(needle,'\nMODES.update('+repr(modes)+')'+needle);path.write_text(s)
    print('TILE_SAFE_CANDIDATES_ADDED',4)


if __name__=='__main__':main()
