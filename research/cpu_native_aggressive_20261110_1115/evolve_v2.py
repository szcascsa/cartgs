"""保留首版后移除内循环Tensor元数据访问；只保留构造时的指针绑定。"""
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
with tarfile.open(HERE/'SOURCE_V1.tar.gz','x:gz') as archive:
    for name in ['source/cpu_native.cpp','cpu_runtime.py']:archive.add(HERE/name,arcname=name)
p=HERE/'source/cpu_native.cpp';s=p.read_text()
s=s.replace('int in=0,out=0,padded=0;T packed,bias;','int in=0,out=0,padded=0;T packed,bias;const float* weights_ptr=nullptr;const float* bias_ptr=nullptr;')
s=s.replace('out=v.size(0);in=v.size(1);padded=', 'out=v.size(0);in=v.size(1);padded=')
needle='    for(int o=0;o<out;++o){bias.data_ptr<float>()[o]=q.data_ptr<float>()[o];for(int j=0;j<in;++j)dst[(o/8)*in*8+j*8+o%8]=src[o*in+j];}'
assert needle in s;s=s.replace(needle,needle+'\n    weights_ptr=packed.data_ptr<float>();bias_ptr=bias.data_ptr<float>();',1)
s=s.replace('const float* weights=packed.data_ptr<float>();const float* b=bias.data_ptr<float>();','const float* weights=weights_ptr;const float* b=bias_ptr;')
s=s.replace('  const float* f(int i)const{return s_[i].data_ptr<float>();}',
    '  const float* F_[13]{};const int64_t* owner_ptr_=nullptr;const bool* eligible_ptr_=nullptr;\n  float *pselector_=nullptr,*poffset_=nullptr,*pfeature_=nullptr,*pprefix_=nullptr;std::vector<const float*> table_ptrs_;\n  const float* f(int i)const{return F_[i];}')
s=s.replace('return s_[2].data_ptr<int64_t>();','return owner_ptr_;').replace('return s_[3].data_ptr<bool>();','return eligible_ptr_;')
needle='    for(int i=0;i<13;++i)s_.push_back(own(state[i],i==2?at::kLong:i==3?at::kBool:at::kFloat));'
assert needle in s;s=s.replace(needle,needle+'\n    for(int i=0;i<13;++i)if(i!=2&&i!=3)F_[i]=s_[i].data_ptr<float>();owner_ptr_=s_[2].data_ptr<int64_t>();eligible_ptr_=s_[3].data_ptr<bool>();',1)
needle='    feature_projected_=at::empty({int64_t(tables.size()),A_,H_},options);feature_prefix_=at::empty_like(feature_projected_);'
assert needle in s
s=s.replace(needle,needle+'\n    pselector_=selector_prefix_data_.data_ptr<float>();poffset_=offset_prefix_.data_ptr<float>();pfeature_=feature_projected_.data_ptr<float>();pprefix_=feature_prefix_.data_ptr<float>();\n    for(auto& t:tables_)table_ptrs_.push_back(t.data_ptr<float>());',1)
for old,new in [('selector_prefix_data_.data_ptr<float>()','pselector_'),('offset_prefix_.data_ptr<float>()','poffset_'),('feature_projected_.data_ptr<float>()','pfeature_'),('feature_prefix_.data_ptr<float>()','pprefix_')]:
    s=s.replace(old,new)
# 恢复构造时唯一一次的真实指针绑定。
s=s.replace('pselector_=pselector_;poffset_=poffset_;pfeature_=pfeature_;pprefix_=pprefix_;',
    'pselector_=selector_prefix_data_.data_ptr<float>();poffset_=offset_prefix_.data_ptr<float>();pfeature_=feature_projected_.data_ptr<float>();pprefix_=feature_prefix_.data_ptr<float>();')
for name in ['sel0_','comp0_']:
    s=s.replace(name+'.packed.data_ptr<float>()',name+'.weights_ptr').replace(name+'.bias.data_ptr<float>()',name+'.bias_ptr')
for name in ['0','l','partial']:s=s.replace('tables_['+name+'].data_ptr<float>()','table_ptrs_['+name+']')
s=s.replace('class CpuPipeline {','class CpuPipelineV2 {').replace('  CpuPipeline(', '  CpuPipelineV2(').replace('pybind11::class_<CpuPipeline>', 'pybind11::class_<CpuPipelineV2>').replace('&CpuPipeline::','&CpuPipelineV2::')
p.write_text(s)
p=HERE/'cpu_runtime.py';s=p.read_text().replace("HERE/'build'","HERE/'build_v2'").replace('mycartgs_cpu_aggressive_v1','mycartgs_cpu_aggressive_v2');p.write_text(s)
print('CPU_V2_RAW_POINTERS_GENERATED')
