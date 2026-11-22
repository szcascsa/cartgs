"""每Gaussian只计算一次log-opacity/成员阈值，复用本帧已结束使用的协方差scratch。"""
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def main():
    root=HERE/'source_log';shutil.copytree(HERE.parent/'source',root);(HERE/'build_log').mkdir()
    path=root/'graph_forward.cu';text=path.read_text()
    text=text.replace('bool Dynamic = false, bool Schedule = false>', 'bool Dynamic = false, bool Schedule = false, bool CacheLog = false>')
    anchor='  tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
  // Cov2D/SH已消费完3D covariance，仅冻结forward复用其scratch；不修改外部预计算协方差。
  if constexpr(CacheLog){cov3Ds[idx*6]=logf(opacities[idx]);cov3Ds[idx*6+1]=logf(opacities[idx]/(1.0f/255.0f));}
''')
    anchor='int threads,uint2* tasks,uint32_t* task_count) {'
    assert text.count(anchor)==1
    text=text.replace(anchor,'int threads,uint2* tasks,uint32_t* task_count,bool cache_log) {')
    start=text.index('  if(tasks) {',text.index('void graph_preprocess'))
    end=text.index('\n}',start)
    original=text[start:end]
    cached=original.replace('preprocessCUDA<3,true,true,false,true,true>', 'preprocessCUDA<3,true,true,false,true,true,true>')
    cached=cached.replace('preprocessCUDA<3,true,true,false,true>', 'preprocessCUDA<3,true,true,false,true,false,true>')
    text=text[:start]+'  if(cache_log){\n'+cached+'\n  }else{\n'+original+'\n  }'+text[end:]
    path.write_text(text)
    path=root/'graph_inputs.h';text=path.read_text().replace('uint32_t* task_count=nullptr);','uint32_t* task_count=nullptr,bool cache_log=false);');path.write_text(text)
    path=root/'preprocess_tasks.cuh';text=path.read_text().replace('template<int ThreadsPerTask>', 'template<int ThreadsPerTask,bool CacheLog=false>')
    text=text.replace('int words,dim3 grid) {','int words,dim3 grid,const float* log_cache=nullptr) {')
    text=text.replace('const float limit=logf(co.w/(1.0f/255.0f));','const float limit=CacheLog?log_cache[id*6+1]:logf(co.w/(1.0f/255.0f));')
    path.write_text(text)
    for original_name,new_name in [('folded_pixels.cuh','log_pixels.cuh'),('diagnosed_pixels.cuh','log_diagnosed_pixels.cuh')]:
        text=(root/original_name).read_text()
        if original_name=='folded_pixels.cuh':
            start=text.index('__device__ __forceinline__ FoldedGaussian foldGaussian(')
            helper=text[start:text.index('template<int Mode>',start)].replace('foldGaussian(', 'foldCachedGaussian(')
            helper=helper.replace('float cx,float cy)', 'float cx,float cy,float cached_log)')
            helper=helper.replace('logf(co.w)', 'cached_log')
            kernel=text[text.index('template<int Mode> __global__ void renderFoldedPixels'):]
            kernel=kernel.replace('renderFoldedPixels','renderLogPixels')
            kernel=kernel.replace('const GraphInputs* inputs) {','const GraphInputs* inputs,const float* log_cache) {')
            kernel=kernel.replace('foldGaussian(xy[id],conic[id],cx,cy)', 'foldCachedGaussian(xy[id],conic[id],cx,cy,log_cache[id*6])')
        else:
            start=text.index('__device__ __forceinline__ DebugFoldedGaussian debugFoldGaussian(')
            helper=text[start:text.index('template<int Mode>',start)].replace('debugFoldGaussian(', 'debugFoldCachedGaussian(')
            helper=helper.replace('float cx,float cy)', 'float cx,float cy,float cached_log)')
            helper=helper.replace('logf(co.w)', 'cached_log')
            kernel=text[text.index('template<int Mode> __global__ void renderDiagnosedPixels'):]
            kernel=kernel.replace('renderDiagnosedPixels','renderLogDiagnosedPixels')
            kernel=kernel.replace('unsigned long long* interactions) {', 'unsigned long long* interactions,const float* log_cache) {')
            kernel=kernel.replace('debugFoldGaussian(xy[id],conic[id],cx,cy)', 'debugFoldCachedGaussian(xy[id],conic[id],cx,cy,log_cache[id*6])')
        (root/new_name).write_text(helper+'\n'+kernel)
    path=root/'graph_main.cu';text=path.read_text()
    text=text.replace('#include "diagnosed_pixels.cuh"','#include "diagnosed_pixels.cuh"\n#include "log_pixels.cuh"\n#include "log_diagnosed_pixels.cuh"')
    text=text.replace('pixel_mode<=58','pixel_mode<=112')
    text=text.replace('preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr);',
        'preprocess_threads_,fused_tasks?tasks:nullptr,fused_tasks?task_count:nullptr,pixel_mode_>=109);')
    anchor='      else if(build_mode_==10)executePreprocessedTasks<32>'
    assert text.count(anchor)==1
    text=text.replace(anchor,'''      else if(build_mode_==10&&(pixel_mode_==111||pixel_mode_==112))executePreprocessedTasks<32,true><<<workers,128,0,capture>>>(tasks,task_count,inverse,xy,conic,radii,bits,words,grid,covariance);
      else if(build_mode_==10)executePreprocessedTasks<32>''')
    anchor=next(line for line in text.splitlines() if 'if(pixel_mode_==58)renderFoldedPixels<40>' in line)
    lines=[];diagnosed=[]
    for mode,kind in [(91,37),(92,38),(93,38),(94,38)]:
        if mode==93:
            lines.append(f'      if(pixel_mode_=={mode+18})renderFoldedPixels<38><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs);')
            diagnosed.append(f'    if(pixel_mode_=={mode+18})renderDiagnosedPixels<38><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()));')
        else:
            lines.append(f'      if(pixel_mode_=={mode+18})renderLogPixels<{kind}><<<grid,dim3(16,8),0,capture>>>(W,H,bits,words,directory,groups,sorted,lists,counts,list_capacity,xy,conic,rgb,nullptr,nullptr,key_out,inputs,covariance);')
            diagnosed.append(f'    if(pixel_mode_=={mode+18})renderLogDiagnosedPixels<{kind}><<<grid,dim3(16,8),0,stream>>>(W,H,bits,words,directory,groups,sorted,lists,counts,p.list_capacity,xy,conic,rgb,nullptr,nullptr,ko,inputs,image.data_ptr<float>(),reinterpret_cast<unsigned*>(flags.data_ptr<int>()),reinterpret_cast<unsigned long long*>(interactions.data_ptr<int64_t>()),covariance);')
    text=text.replace(anchor,anchor+'\n'+'\n'.join(lines)).replace('pixel_mode_>=55&&pixel_mode_<=58','pixel_mode_>=55&&pixel_mode_<=112')
    anchor='    C10_CUDA_KERNEL_LAUNCH_CHECK();return {image,flags,interactions};'
    text=text.replace(anchor,'\n'.join(diagnosed)+'\n'+anchor).replace('NineHourRasterizer','StudentLogRasterizer')
    path.write_text(text)
    print('STUDENT_LOG_CACHE_SOURCE_READY')


if __name__=='__main__':main()
