"""从冻结V4生成独立候选：调度融合/AVX2微核分别开关，不改旧目录。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'cpu_native_aggressive_20261110_1115'
source=(OLD/'source/cpu_native.cpp').read_text(encoding='utf-8').replace('CpuPipelineV4','CpuPipelineDeep')
def change(before,after):
    global source
    assert source.count(before)==1,(before[:120],source.count(before));source=source.replace(before,after)

change('int in=0,out=0,padded=0;','int in=0,out=0,padded=0;int fast=0;')
change('  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{',
    (HERE/'source/dense_fast.h').read_text(encoding='utf-8')+'''\n  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{
    if(fast==1&&out%16==0){
      if(rows==4)fast_rows<4>(x,xs,y,ys,activation);else if(rows==3)fast_rows<3>(x,xs,y,ys,activation);
      else if(rows==2)fast_rows<2>(x,xs,y,ys,activation);else fast_rows<1>(x,xs,y,ys,activation);return;
    }''')
change('  Pool pool_;std::mutex call_;','  bool fused_schedule_;int kernel_mode_;\n  Pool pool_;std::mutex call_;')
change('bool selector_prefix=true,bool factored=true,bool diagnose=false)\n      :pool_(threads)',
    'bool selector_prefix=true,bool factored=true,bool diagnose=false,bool fused_schedule=false,int kernel_mode=0)\n      :fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),pool_(threads)')
change('    activated_scale_.resize(N_*3);activated_opacity_.resize(N_);',
    '''    TORCH_CHECK(kernel_mode>=0&&kernel_mode<=1,"Unknown AVX2 kernel mode");
    sel0_.fast=sel1_.fast=sel2_.fast=comp0_.fast=comp1_.fast=ratio_.fast=combined_head_.fast=kernel_mode;
    for(auto& h:ho_)h.fast=kernel_mode;
    activated_scale_.resize(N_*3);activated_opacity_.resize(N_);''')
change('    if(schema_)view_=Dense(view_weight,at::zeros({H_},view_weight.options()));',
    '    if(schema_){view_=Dense(view_weight,at::zeros({H_},view_weight.options()));view_.fast=kernel_mode;}')
# 用同一helper执行原Anchor动态公式，融合候选在视锥mask写出后立即计算。
anchor='''  void anchor_context(int a,const float* vm,const float* pm,const float* camera,float aspect,float near,bool local,bool compensation){
    bool selector_real=schema_==2||schema_==3,comp_real=schema_==2||schema_==4;
    if(local&&schema_&&(selector_real||(compensation&&comp_real)))encode_view(a,vm,pm,camera,aspect,near);
    if(schema_){alignas(32) float x[8];for(int j=0;j<7;++j)x[j]=local&&selector_real?view_code_[a*7+j]:0.f;x[7]=local?1.f:0.f;
      for(int o=0;o<32;o+=8){__m256 v=_mm256_setzero_ps();for(int j=0;j<8;++j)v=_mm256_fmadd_ps(_mm256_set1_ps(x[j]),_mm256_loadu_ps(sel0_.weights_ptr+(o/8)*sel0_.in*8+(j+11)*8),v);_mm256_storeu_ps(sel_dynamic_.data()+a*32+o,v);}}
  }
'''
change('  void frustum(const float* pm,bool global){',anchor+'''  void frustum(const float* pm,bool global,const float* vm,const float* camera,float aspect,float near,bool local,bool compensation,bool context){''')
change('if(global){for(int j=a;j<std::min(a+8,A_);++j)visible_[j]=occupied_[j];continue;}',
    'if(global){for(int j=a;j<std::min(a+8,A_);++j){visible_[j]=occupied_[j];if(context&&visible_[j])anchor_context(j,vm,pm,camera,aspect,near,local,compensation);}continue;}')
change('visible_[a]=keep;}\n      }','visible_[a]=keep;}\n        if(context)for(int j=block*8;j<std::min(block*8+8,A_);++j)if(visible_[j])anchor_context(j,vm,pm,camera,aspect,near,local,compensation);\n      }')
start=source.index('      pool_.run(A_,128,[&](int begin,int end,int){alignas(32) float x[8],result[32];')
end=source.index('checkpoint();',start)+len('checkpoint();')
source=source[:start]+'''      if(!fused_schedule_)pool_.run(A_,128,[&](int begin,int end,int){for(int a=begin;a<end;++a)if(visible_[a])anchor_context(a,vm.data_ptr<float>(),pm.data_ptr<float>(),camera.data_ptr<float>(),float(width)/height,near,local,compensation);});checkpoint();'''+source[end:]
change('    frustum(pm.data_ptr<float>(),global);candidates(global);',
    '''    bool context=fused_schedule_&&kind!=3&&((kind==1&&budget>0)||(kind!=1&&ratio>0));
    frustum(pm.data_ptr<float>(),global,vm.data_ptr<float>(),camera.data_ptr<float>(),float(width)/height,near,local,compensation,context);candidates(global);''')
change('scores_[base+r]=eb/(ea+eb);','scores_[base+r]=eb/(ea+eb);if(fused_schedule_)keys_[base+r]={eligible()[id]?scores_[base+r]:-INFINITY,uint32_t(id)};')
change('      selector(float(std::log(ratio)),local,selector_real);','      keys_.resize(M_);selector(float(std::log(ratio)),local,selector_real);')
change('      keys_.resize(M_);pool_.run(M_,1024,','      if(!fused_schedule_)pool_.run(M_,1024,')
# 一次扫描同时生成各线程候选，随后按线程原行序向前压缩；无需第二次线程池分发。
change('    std::fill(counts_.begin(),counts_.end(),0);',
    '''    if(fused_schedule_){
      pool_.run(N_,0,[&](int begin,int end,int thread){int n=0;for(int i=begin;i<end;++i)if(visible_[owners()[i]])rows_[begin+n++]=i;counts_[thread]=n;});
      M_=0;for(int thread=0;thread<pool_.count();++thread){int begin=int(int64_t(N_)*thread/pool_.count()),n=counts_[thread];std::memmove(rows_.data()+M_,rows_.data()+begin,n*sizeof(int64_t));M_+=n;}return;
    }
    std::fill(counts_.begin(),counts_.end(),0);''')
# 稳定Softmax中最大项exp(0)恒为1；只省去该调用，不用近似exp或改变加法顺序。
change('float ea=std::exp(a-max),eb=std::exp(b-max);',
    'float ea,eb;if(fused_schedule_&&std::isfinite(a)&&std::isfinite(b)){if(a>=b){ea=1.f;eb=std::exp(b-a);}else{ea=std::exp(a-b);eb=1.f;}}else{ea=std::exp(a-max);eb=std::exp(b-max);}')
source=source.replace('#include <torch/extension.h>','#include <torch/extension.h>\n#include <cstring>')
# C++创建属性视图、stats字典和紧凑相机包，直接供CPU与H2D使用。
methods='''  std::vector<T> packet_views(const T& flat,int64_t k)const{
    TORCH_CHECK(flat.dim()==1&&flat.numel()==k*width_,"Packet size");
    std::vector<T> v;int64_t p=0;
    for(int w:{3,1,3,4,3,rest_,3}){auto t=flat.narrow(0,p,k*w);p+=k*w;
      if(v.size()==4)v.push_back(t.view({k,1,3}));else if(v.size()==5)v.push_back(t.view({k,rest_/3,3}));else v.push_back(t.view({k,w}));}
    return v;
  }
  int64_t total()const{return N_;}int64_t width()const{return width_;}
'''
change('  int64_t bytes()const{return bytes_;}',methods+'  int64_t bytes()const{return bytes_;}')
change('pybind11::class_<CpuPipelineDeep>(m,"CpuPipeline")','pybind11::class_<CpuPipelineDeep,std::shared_ptr<CpuPipelineDeep>>(m,"CpuPipeline")')
change('int,float,bool,int,bool,bool,bool>()','int,float,bool,int,bool,bool,bool,bool,int>()')
change('pybind11::arg("diagnose")=false)','pybind11::arg("diagnose")=false,pybind11::arg("fused_schedule")=false,pybind11::arg("kernel_mode")=0)')
change('    .def("diagnostics",&CpuPipelineDeep::diagnostics)',
    '    .def("packet_views",&CpuPipelineDeep::packet_views).def("diagnostics",&CpuPipelineDeep::diagnostics)')
# 原生交接头在同一个扩展中，GPU只进行复制和Event管理，不执行网络。
source=source.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){','#include "native_handoff.h"\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){')
change('    .def("diagnostics",&CpuPipelineDeep::diagnostics)', '    .def("diagnostics",&CpuPipelineDeep::diagnostics)') if False else None
source=source.replace('.def("workspace_bytes",&CpuPipelineDeep::bytes);','.def("workspace_bytes",&CpuPipelineDeep::bytes);\n  bind_handoff(m);')
source=source.replace('    .def("packet_views",&CpuPipelineDeep::packet_views)',
    '''    .def("prepare_packet",&prepare_cpu,pybind11::arg("camera"),pybind11::arg("kind"),pybind11::arg("ratio"),pybind11::arg("budget"),pybind11::arg("local"),pybind11::arg("compensation"),pybind11::arg("timing"),pybind11::arg("output"),pybind11::arg("id_output"),pybind11::arg("sh_degree"))
    .def("packet_views",&CpuPipelineDeep::packet_views)''')
(HERE/'source/cpu_deep.cpp').write_text(source,encoding='utf-8',newline='\n')
print('DEEP_SOURCE_GENERATED',len(source))
