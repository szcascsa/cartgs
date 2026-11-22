"""V4：保持独立四Head权重，合并相同输入的隐藏层调度；静态激活只计算一次。"""
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
with tarfile.open(HERE/'SOURCE_V3.tar.gz','x:gz') as archive:
    for n in ['source/cpu_native.cpp','cpu_runtime.py']:archive.add(HERE/n,arcname=n)
p=HERE/'source/cpu_native.cpp';s=p.read_text()
s=s.replace('std::vector<Dense> hh_,ho_;','std::vector<Dense> hh_,ho_;Dense combined_head_;std::vector<float> activated_scale_,activated_opacity_;')
s=s.replace('float base=std::exp(f(5)[id*3+j]),value=base;','float base=activated_scale_[id*3+j],value=base;',1)
s=s.replace('float logit=f(7)[id]+(compensate?delta[10]:0.f);op[row]=1.f/(1.f+std::exp(-logit));',
    'float logit=f(7)[id]+(compensate?delta[10]:0.f);op[row]=compensate?1.f/(1.f+std::exp(-logit)):activated_opacity_[id];',1)
s=s.replace('float h0[4][64],h1[4][64],hidden[4][64],small[4][4]', 'float h0[4][64],h1[4][64],hidden[4][256],small[4][4]',1)
old='''        for(int head=0;head<4;++head){const float* source=h1[0];if(!student_){hh_[head].apply(source,R,64,hidden[0],64,true);source=hidden[0];}
          ho_[head].apply(source,R,64,small[0],4);'''
new='''        if(!student_)combined_head_.apply(h1[0],R,64,hidden[0],256,true);
        for(int head=0;head<4;++head){const float* source=student_?h1[0]:hidden[0]+head*64;
          ho_[head].apply(source,R,student_?64:256,small[0],4);'''
assert old in s;s=s.replace(old,new,1)
needle='    if(schema_)view_=Dense(view_weight,at::zeros({H_},view_weight.options()));'
assert needle in s
s=s.replace(needle,'''    if(!student_){std::vector<T> ws,bs;for(int head=0;head<4;++head){ws.push_back(comp[5+head*4]);bs.push_back(comp[6+head*4]);}combined_head_=Dense(at::cat(ws,0),at::cat(bs,0));}
    activated_scale_.resize(N_*3);activated_opacity_.resize(N_);
    pool_.run(N_,128,[&](int begin,int end,int){for(int id=begin;id<end;++id){for(int j=0;j<3;++j)activated_scale_[id*3+j]=std::exp(f(5)[id*3+j]);activated_opacity_[id]=1.f/(1.f+std::exp(-f(7)[id]));}});
'''+needle,1)
s=s.replace('    bytes_+=int64_t(N_)*(sizeof(int64_t)*2+sizeof(Key))',
    '    bytes_+=int64_t(activated_scale_.size()+activated_opacity_.size())*4;\n    bytes_+=int64_t(N_)*(sizeof(int64_t)*2+sizeof(Key))',1)
s=s.replace('CpuPipelineV3','CpuPipelineV4');p.write_text(s)
p=HERE/'cpu_runtime.py';s=p.read_text().replace('build_v3','build_v4').replace('mycartgs_cpu_aggressive_v3','mycartgs_cpu_aggressive_v4');p.write_text(s)
print('CPU_V4_GROUPED_HEAD_GENERATED')
