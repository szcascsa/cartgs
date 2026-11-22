"""显式固定批长合成循环，减少逐成员列表边界检查；成员与像素次序不变。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_chunk_loop';shutil.copytree(HERE/'source_encoded_sign',root);(HERE/'build_chunk_loop').mkdir()
    for name in ['encoded_sign_pixels.cuh','encoded_sign_diagnosed.cuh']:
        text=(root/name).read_text();text=text[text.index('template<int Mode,int Kind,int Layout>'):]
        text=text.replace('template<int Mode,int Kind,int Layout>','template<int Mode,int Kind,int Layout,int Chunk>')
        text=text.replace('renderEncodedSign','renderChunkLoop')
        start=text.index('    #pragma unroll 32\n    for(int j=0;j<count;++j){')
        end=text.index('\n    }\n  }\n  } else',start)+len('\n    }')
        original=text[start:end]
        body=original[original.index('      if constexpr(Kind<3)'):original.rindex('\n    }')]
        # 完整批与最后不足一批均按原有有序成员发布。
        full_body=body.replace('Kind,Step','Kind,Step').replace('[j]','[j+k]').replace('+j]','+j+k]')
        assert 'j+k' in full_body
        new='''    int j=0;
    for(;j+Chunk<=count;j+=Chunk){
      if(__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
      #pragma unroll
      for(int k=0;k<Chunk;++k){
'''+full_body+'''
      }
    }
    for(;j<count;++j){
      if(__all_sync(0xffffffff,T[0]<0.f&&T[1]<0.f))break;
'''+body+'''
    }'''
        text=text[:start]+new+text[end:]
        (root/name.replace('encoded_sign','chunk_loop')).write_text(text)
    path=root/'graph_main.cu';text=path.read_text().replace('#include "encoded_sign_diagnosed.cuh"',
        '#include "encoded_sign_diagnosed.cuh"\n#include "chunk_loop_pixels.cuh"\n#include "chunk_loop_diagnosed.cuh"')
    text=text.replace('pixel_mode<=202','pixel_mode<=213').replace('pixel_mode_<=202','pixel_mode_<=213')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==202)renderEncodedSignPixels' in line)
    launches=[];diagnostics=[]
    for mode,chunk in [(191,2),(192,4),(193,8),(194,16),(195,32)]:
        launches.append(f'      if(pixel_mode_=={mode+18})renderChunkLoopPixels<38,3,1,{chunk}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
        diagnostics.append(f'    if(pixel_mode_=={mode+18})renderChunkLoopDiagnosedPixels<38,3,1,{chunk}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(launches))
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnostics)+'\n'+anchor).replace('StudentEncodedSignRasterizer','StudentChunkLoopRasterizer');path.write_text(text)
    print('CHUNK_LOOP_SOURCE_READY')


if __name__=='__main__':main()
