"""局部折叠的异常属性回退及颜色公共权重候选，保留上版完整so。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'reports/DEV_CONTROLLER03.json').read_text())['state']=='complete'
    snapshot=HERE/'development_v2';snapshot.mkdir();shutil.copytree(HERE/'source',snapshot/'source')
    for name,origin in [('mycartgs_footprint_ninehour_v1.so',HERE/'build/mycartgs_footprint_ninehour_v1.so'),('render_runtime.py',HERE/'render_runtime.py')]:shutil.copy2(origin,snapshot/name)
    path=HERE/'source/folded_pixels.cuh';s=path.read_text()
    s=s.replace('Mode==19||Mode==20?1:','Mode==19||Mode==20||Mode>=28&&Mode<=31?1:')
    s=s.replace('C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;T=next;',
        'if(Mode==30||Mode==31||Mode==32){const float w=alpha*T;C[0]=fmaf(color.x,w,C[0]);C[1]=fmaf(color.y,w,C[1]);C[2]=fmaf(color.z,w,C[2]);}\n  else {C[0]+=color.x*alpha*T;C[1]+=color.y*alpha*T;C[2]+=color.z*alpha*T;}T=next;')
    s=s.replace('Mode==16||Mode==20||Mode==22?.01f:','Mode==16||Mode==20||Mode==22||Mode==29||Mode==31?.01f:')
    s=s.replace('__shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256];',
        '__shared__ float4 abc[256];__shared__ float2 ef[256];__shared__ float3 rgb[256];__shared__ unsigned valid[Mode>=28?256:1];')
    s=s.replace('abc[j]=f.a;ef[j]=f.b;rgb[j]=', '''if(Mode>=28){
        const float4 co=conic[id];bool safe=co.w>0.f&&co.w<=1.f;
        const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
        for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
        valid[j]=safe;
        if(!safe){f.a={xy[id].x,xy[id].y,co.x,co.y};f.b={co.z,co.w};}
      }
      abc[j]=f.a;ef[j]=f.b;rgb[j]=''')
    s=s.replace('foldedPair<Mode>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C);', '''if(Mode>=28&&!valid[j]){
        const float4 a=abc[j];const float2 b=ef[j];
        float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
        compositeScalingPair<9>({a.x,a.y},{a.z,a.w,b.x,b.y},rgb[j],pix,done,T,C,-CUDART_INF_F);
      }else foldedPair<Mode>({abc[j],ef[j]},rgb[j],lx,ly,done,T,C);''')
    old='''auto f=foldGaussian(xy[id],conic[id],cx,cy);
          foldedPair<Mode>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C);'''
    new='''auto f=foldGaussian(xy[id],conic[id],cx,cy);
          bool safe=conic[id].w>0.f&&conic[id].w<=1.f;
          const float values[6]={f.a.x,f.a.y,f.a.z,f.a.w,f.b.x,f.b.y};
          for(int v=0;v<6;++v)safe=safe&&isfinite(values[v])&&fabsf(values[v])<10000.f;
          if(Mode>=28&&!safe){
            float2 pix[2]={{float(x),float(y)},{float(x),float(y+8)}};
            compositeScalingPair<9>(xy[id],conic[id],{colors[id*3],colors[id*3+1],colors[id*3+2]},pix,done,T,C,-CUDART_INF_F);
          }else foldedPair<Mode>(f,{colors[id*3],colors[id*3+1],colors[id*3+2]},lx,ly,done,T,C);'''
    assert old in s;s=s.replace(old,new);path.write_text(s)
    path=HERE/'source/graph_main.cu';s=path.read_text().replace('pixel_mode<=45','pixel_mode<=50')
    anchor=next(line for line in s.splitlines() if 'if(pixel_mode_==45)renderApproxPixels<27>' in line)
    extra='\n'.join(f'      if(pixel_mode_=={18+m})renderFoldedPixels<{m}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);' for m in range(28,33))
    s=s.replace(anchor,anchor+'\n'+extra);path.write_text(s)
    path=HERE/'render_runtime.py';s=path.read_text();needle='\n\n\n@lru_cache'
    modes={28:'folded_bitexp_safe',29:'folded_bitexp_stop_safe',30:'folded_bitexp_weight',31:'folded_bitexp_weight_stop',32:'folded_fast_weight'}
    s=s.replace(needle,'\nMODES.update('+repr(modes)+')'+needle);path.write_text(s)
    print('SAFE_FOLDED_ADDED',len(modes))


if __name__=='__main__':main()
