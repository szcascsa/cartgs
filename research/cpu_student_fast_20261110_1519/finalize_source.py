"""撤回未获稳定收益的SoA、私有Anchor缓存、固定循环界与rsqrt。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=HERE/'source/cpu_student_fast.cpp';s=p.read_text(encoding='utf-8')
start=s.index('template<int Rows,int Outputs>\ninline void apply_student32');end=s.index('template<int Rows> inline void fast_rows',start);s=s[:start]+s[end:]
s=s.replace('Dense combined_head_,student_head_;T student_body_plain_,student_head_plain_;int student_kernel_,student_decode_,student_context_;uint32_t student_stamp_=0;std::vector<float> student_context_cache_;std::vector<uint32_t> student_context_marks_;','Dense combined_head_,student_head_;int student_kernel_,student_decode_,student_context_;')
start=s.index('  void student_context_fill(');end=s.index('  void comp_context(',start);s=s[:start]+s[end:]
s=s.replace('if(student_kernel_==3)student_soa(ratio,local,real,output);else student_compact(ratio,local,real,output);','student_compact(ratio,local,real,output);')
s=s.replace('student_kernel<=4&&student_decode>=0&&student_decode<=2','student_kernel<=2&&student_decode>=0&&student_decode<=1').replace('student_context<=2','student_context<=1')
start=s.index('    if(student_&&student_context_==2){');end=s.index('\n    if(student_){TORCH_CHECK(H_==32',start);s=s[:start]+s[end:]
s=s.replace('student_body_plain_=own(comp[2]);student_head_plain_=own(weight);','')
s=s.replace('    bytes_+=student_context_cache_.size()*sizeof(float)+student_context_marks_.size()*sizeof(uint32_t);\n','').replace('student_body_plain_.nbytes()+student_head_plain_.nbytes()+','')
assert 'student_soa' not in s and 'student_context_cache_' not in s and 'student_fixed' not in s
p.write_text(s,encoding='utf-8')
p=HERE/'source/student_fast.h';s=p.read_text(encoding='utf-8');start=s.index('// 仅冻结32/32学生。');s=s[start:]
start=s.index('// 8点SoA微核');end=s.index('void student_publish_vector(',start);s=s[:start]+s[end:]
s=s.replace('  int full=0,partial=-1;float weight=0;alignas(32) float rp[32];student_context_begin(ratio,local,real,full,partial,weight,rp);','  comp_context(ratio,local,real);')
s=s.replace('const float* dynamic=student_dynamic(a,worker,full,partial,weight,rp,local,real);','const float* dynamic=comp_dynamic_.data()+a*32;')
start=s.index('      if(student_kernel_==4){');end=s.index('\n      for(int r=0;r<R;++r){int id=selected_[base+r];float reference=',start)
s=s[:start]+'      comp1_.apply(h0[0],R,32,h1[0],32,true);student_head_.apply(h1[0],R,32,delta[0],16);'+s[end:]
start=s.index('  auto norm4=');end=s.index('\n  __m256 d[4]',start)
s=s[:start]+'''  auto norm4=[&](__m256* q){__m256 s=_mm256_mul_ps(q[0],q[0]);for(int j=1;j<4;++j)s=_mm256_add_ps(s,_mm256_mul_ps(q[j],q[j]));s=_mm256_max_ps(floor,_mm256_sqrt_ps(s));for(int j=0;j<4;++j)q[j]=_mm256_div_ps(q[j],s);};'''+s[end:]
assert 'student_soa' not in s and 'student_mlp32' not in s and 'rsqrt' not in s
p.write_text(s,encoding='utf-8')
p=HERE/'common.py';s=p.read_text(encoding='utf-8');start=s.index('    modes=dict(');end=s.index('\n    kernel,decode,context=',start)
s=s[:start]+'''    modes=dict(control=(0,0,0),heads=(1,0,0),compact=(2,0,0),vector=(2,1,0),context=(2,1,1))'''+s[end:];p.write_text(s,encoding='utf-8')
p=HERE/'student_fast_runtime.py';s=p.read_text(encoding='utf-8').replace('student_checkpoint=None,**kwargs):','student_checkpoint=None,student_kernel=2,student_decode=1,student_context=1,**kwargs):').replace('student=student,**kwargs)','student=student,student_kernel=student_kernel,student_decode=student_decode,student_context=student_context,**kwargs)');p.write_text(s,encoding='utf-8')
print('ONLY_VALIDATED_CANDIDATE_PATHS_RETAINED')
