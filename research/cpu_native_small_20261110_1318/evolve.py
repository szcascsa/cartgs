"""由上一轮冻结源码生成独立小输出、顺序压缩与线程池候选。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'cpu_native_deep_20261110_1222'
source=(OLD/'source/cpu_deep.cpp').read_text(encoding='utf-8').replace('CpuPipelineDeep','CpuPipelineSmall')
def change(before,after):
    global source
    assert source.count(before)==1,(before[:100],source.count(before));source=source.replace(before,after)

change('int in=0,out=0,padded=0;int fast=0;','int in=0,out=0,padded=0;int fast=0;bool small=false;')
change('  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{',
    (HERE/'source/dense_small.h').read_text(encoding='utf-8')+'''\n  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{
    if(small&&out>=2&&out<=4&&(in==32||in==64)){
      if(rows==4)small_rows<4>(x,xs,y,ys,activation);else if(rows==3)small_rows<3>(x,xs,y,ys,activation);
      else if(rows==2)small_rows<2>(x,xs,y,ys,activation);else small_rows<1>(x,xs,y,ys,activation);return;
    }''')
change('  bool fused_schedule_;int kernel_mode_;','  bool fused_schedule_;int kernel_mode_,ordered_mode_;')
change('bool fused_schedule=false,int kernel_mode=0)', 'bool fused_schedule=false,int kernel_mode=0,bool small_kernel=false,int ordered_mode=0)')
change(':fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),pool_(threads)',':fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),ordered_mode_(ordered_mode),pool_(threads)')
change('    for(auto& h:ho_)h.fast=kernel_mode;',
    '''    for(auto& h:ho_){h.fast=kernel_mode;h.small=small_kernel;}sel2_.small=small_kernel;
    TORCH_CHECK(ordered_mode>=0&&ordered_mode<=3,"Small kernel/ordered mode");''')
# 原Score与UID唯一复合排序的第K+1名作为边界，直接恢复原行号顺序。
ordered='''  void ordered_compact(const Key& cutoff,bool global,float threshold,int expected){
    selected_.resize(M_);auto keep=[&](int i){int id=rows_[i];return eligible()[id]&&(global?scores_[i]>=threshold:Better{}(Key{scores_[i],uint32_t(id)},cutoff));};
    int count=0;
    if(ordered_mode_==1){for(int i=0;i<M_;++i)if(keep(i))selected_[count++]=rows_[i];}
    else{
      pool_.run(M_,0,[&](int begin,int end,int thread){int n=0;for(int i=begin;i<end;++i)if(keep(i))selected_[begin+n++]=rows_[i];counts_[thread]=n;});
      for(int thread=0;thread<pool_.count();++thread){int begin=int(int64_t(M_)*thread/pool_.count()),n=counts_[thread];std::memmove(selected_.data()+count,selected_.data()+begin,n*sizeof(int64_t));count+=n;}
    }
    TORCH_CHECK(global||count==expected,"Ordered compaction violated strict K");K_=count;selected_.resize(count);
  }
'''
change('  void controls(double ratio,',ordered+'  void controls(double ratio,')
before='''        for(auto& k:keys_)if(eligible()[k.id]&&k.score>=threshold)selected_.push_back(k.id);K_=selected_.size();
      }else{if(K_<int(keys_.size()))std::nth_element(keys_.begin(),keys_.begin()+K_,keys_.end(),Better{});for(int i=0;i<K_;++i)selected_.push_back(keys_[i].id);}
      std::sort(selected_.begin(),selected_.end());checkpoint();'''
after='''        if(ordered_mode_)ordered_compact(Key{},true,threshold,-1);
        else{for(auto& k:keys_)if(eligible()[k.id]&&k.score>=threshold)selected_.push_back(k.id);K_=selected_.size();}
      }else{
        if(K_<int(keys_.size()))std::nth_element(keys_.begin(),keys_.begin()+K_,keys_.end(),Better{});
        bool compact=ordered_mode_&&(ordered_mode_!=3||double(K_)*std::log2(std::max(2,K_))>M_+4096);
        if(compact)ordered_compact(K_<M_?keys_[K_]:Key{-INFINITY,uint32_t(-1)},false,0,K_);
        else{for(int i=0;i<K_;++i)selected_.push_back(keys_[i].id);if(ordered_mode_)std::sort(selected_.begin(),selected_.end());}
      }
      if(!ordered_mode_)std::sort(selected_.begin(),selected_.end());checkpoint();'''
change(before,after)
change('int,float,bool,int,bool,bool,bool,bool,int>()','int,float,bool,int,bool,bool,bool,bool,int,bool,int>()')
change('pybind11::arg("kernel_mode")=0)', 'pybind11::arg("kernel_mode")=0,pybind11::arg("small_kernel")=false,pybind11::arg("ordered_mode")=0)')
(HERE/'source/cpu_small.cpp').write_text(source,encoding='utf-8',newline='\n')
header=(OLD/'source/native_handoff.h').read_text(encoding='utf-8').replace('CpuPipelineDeep','CpuPipelineSmall').replace('NativeHandoff','NativeHandoffSmall').replace('NativeTicket','NativeTicketSmall').replace('"NativeHandoffSmall"','"NativeHandoff"')
(HERE/'source/native_handoff.h').write_text(header,encoding='utf-8',newline='\n');print('SMALL_SOURCE',len(source))
