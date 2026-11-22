"""由上一轮冻结源码生成独立小输出、顺序压缩与线程池候选。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'cpu_native_deep_20261110_1222'
source=(OLD/'source/cpu_deep.cpp').read_text(encoding='utf-8').replace('CpuPipelineDeep','CpuPipelineSmall')
def change(before,after):
    global source
    assert source.count(before)==1,(before[:100],source.count(before));source=source.replace(before,after)

change('int in=0,out=0,padded=0;int fast=0;','int in=0,out=0,padded=0;int fast=0,small=0;')
change('  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{',
    (HERE/'source/dense_small.h').read_text(encoding='utf-8')+'''\n  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{
    if(small&&out>=2&&out<=4&&(in==32||in==64)){
      if(rows==4)small_rows<4>(x,xs,y,ys,activation);else if(rows==3)small_rows<3>(x,xs,y,ys,activation);
      else if(rows==2)small_rows<2>(x,xs,y,ys,activation);else small_rows<1>(x,xs,y,ys,activation);return;
    }''')
change('  bool fused_schedule_;int kernel_mode_;','  bool fused_schedule_;int kernel_mode_,ordered_mode_;')
change('bool fused_schedule=false,int kernel_mode=0)', 'bool fused_schedule=false,int kernel_mode=0,int small_kernel=0,int ordered_mode=0,int pool_policy=0)')
change(':fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),pool_(threads)',':fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),ordered_mode_(ordered_mode),pool_(threads,pool_policy)')
change('    for(auto& h:ho_)h.fast=kernel_mode;',
    '''    for(auto& h:ho_){h.fast=kernel_mode;h.small=small_kernel;}sel2_.small=small_kernel;
    TORCH_CHECK(ordered_mode>=0&&ordered_mode<=3&&small_kernel>=0&&small_kernel<=2,"Small kernel/ordered mode");''')
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
# 只在诊断模式采集CPU/墙钟/迁移；最终计时关闭trace。
change('#include <vector>','#include <vector>\n#include <sched.h>\n#include <pthread.h>\n#include <time.h>')
change('  int count_;std::vector<std::thread>', '  int count_,policy_;std::vector<std::thread>')
change('std::atomic<uint64_t> generation_{0};std::atomic<int> remaining_{0};std::atomic<bool> stopping_{false};',
    'alignas(64) std::atomic<uint64_t> generation_{0};alignas(64) std::atomic<int> remaining_{0};alignas(64) std::atomic<bool> stopping_{false};')
change('  void part(int id){int begin=int(int64_t(size_)*id/count_),end=int(int64_t(size_)*(id+1)/count_);function_(begin,end,id);}',
    '''  struct alignas(64) Timing {double wall=0,cpu=0,delay=0;int before=-1,after=-1;};
  std::vector<Timing> timing_;bool trace_=false;Clock::time_point submitted_;std::vector<double> traces_;
  static double cpu_ms(){timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
  void pin(int id){if(policy_!=2)return;cpu_set_t available;CPU_ZERO(&available);if(pthread_getaffinity_np(pthread_self(),sizeof(available),&available))return;
    int n=0;for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&available)){if(n++==id){cpu_set_t one;CPU_ZERO(&one);CPU_SET(cpu,&one);pthread_setaffinity_np(pthread_self(),sizeof(one),&one);return;}}}
  void part(int id){int begin=int(int64_t(size_)*id/count_),end=int(int64_t(size_)*(id+1)/count_);
    if(!trace_){function_(begin,end,id);return;}
    auto now=Clock::now();double cpu=cpu_ms();auto& t=timing_[id];t.delay=ms(submitted_,now);t.before=sched_getcpu();
    function_(begin,end,id);t.wall=ms(now,Clock::now());t.cpu=cpu_ms()-cpu;t.after=sched_getcpu();}
''')
change('  void worker(int id){\n    uint64_t seen=0;', '  void worker(int id){\n    pin(id);uint64_t seen=0;')
assert source.count('spins++<12000)_mm_pause();')==2
source=source.replace('spins++<12000)_mm_pause();','spins++<((policy_==1||policy_==2)?2000:12000))_mm_pause();')
change('  explicit Pool(int count):count_(count){','  explicit Pool(int count,int policy=0):count_(count),policy_(policy),timing_(count){TORCH_CHECK(policy>=0&&policy<=3,"Pool policy");')
source=source.replace('if(policy_!=2)return;','if(policy_!=2&&policy_!=3)return;')
change('  int count()const{return count_;}', '''  struct MainAffinity{
    cpu_set_t previous;bool restore=false;
    explicit MainAffinity(int policy){if(policy!=3)return;CPU_ZERO(&previous);if(pthread_getaffinity_np(pthread_self(),sizeof(previous),&previous))return;
      for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&previous)){cpu_set_t one;CPU_ZERO(&one);CPU_SET(cpu,&one);restore=!pthread_setaffinity_np(pthread_self(),sizeof(one),&one);break;}}
    ~MainAffinity(){if(restore)pthread_setaffinity_np(pthread_self(),sizeof(previous),&previous);}
  };
  MainAffinity main_affinity(){return MainAffinity(policy_);}
  int count()const{return count_;}''')
change('    std::lock_guard<std::mutex> lock(call_);auto t0=Clock::now();','    std::lock_guard<std::mutex> lock(call_);auto affinity=pool_.main_affinity();auto t0=Clock::now();')
change('    function_=f;size_=n;remaining_.store(count_-1,std::memory_order_relaxed);','    function_=f;size_=n;if(trace_)submitted_=Clock::now();remaining_.store(count_-1,std::memory_order_relaxed);')
change('spins++<12000)_mm_pause();\n    if(remaining_', 'spins++<(policy_?2000:12000))_mm_pause();\n    if(remaining_') if False else None
change('    if(remaining_.load(std::memory_order_acquire)){std::unique_lock<std::mutex> lock(mutex_);done_.wait(lock,[&]{return remaining_.load(std::memory_order_acquire)==0;});}\n  }',
    '''    if(remaining_.load(std::memory_order_acquire)){std::unique_lock<std::mutex> lock(mutex_);done_.wait(lock,[&]{return remaining_.load(std::memory_order_acquire)==0;});}
    if(trace_&&traces_.size()+7*count_<=65536*7)for(int i=0;i<count_;++i){auto& t=timing_[i];for(double value:{double(n),double(i),t.delay,t.wall,t.cpu,double(t.before),double(t.after)})traces_.push_back(value);}
  }
  void trace(bool enabled){trace_=enabled;traces_.clear();}std::vector<double> traces(){auto result=std::move(traces_);traces_.clear();return result;}''')
change('  int64_t total()const{return N_;}', '  void pool_trace(bool enabled){pool_.trace(enabled);}std::vector<double> pool_traces(){return pool_.traces();}\n  int64_t total()const{return N_;}')
change('int,float,bool,int,bool,bool,bool,bool,int>()','int,float,bool,int,bool,bool,bool,bool,int,int,int,int>()')
change('pybind11::arg("kernel_mode")=0)', 'pybind11::arg("kernel_mode")=0,pybind11::arg("small_kernel")=0,pybind11::arg("ordered_mode")=0,pybind11::arg("pool_policy")=0)')
change('    .def("packet_views",&CpuPipelineSmall::packet_views)', '    .def("pool_trace",&CpuPipelineSmall::pool_trace).def("pool_traces",&CpuPipelineSmall::pool_traces)\n    .def("packet_views",&CpuPipelineSmall::packet_views)')
(HERE/'source/cpu_small.cpp').write_text(source,encoding='utf-8',newline='\n')
header=(OLD/'source/native_handoff.h').read_text(encoding='utf-8').replace('CpuPipelineDeep','CpuPipelineSmall').replace('NativeHandoff','NativeHandoffSmall').replace('NativeTicket','NativeTicketSmall').replace('"NativeHandoffSmall"','"NativeHandoff"')
(HERE/'source/native_handoff.h').write_text(header,encoding='utf-8',newline='\n');print('SMALL_SOURCE',len(source))
