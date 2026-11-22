"""在未冻结开发候选上增加独立Anchor上下文实验，旧版本另行存档。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'source/cpu_student_fast.cpp';s=p.read_text(encoding='utf-8')
def swap(a,b):
    global s
    assert s.count(a)==1,(a[:75],s.count(a));s=s.replace(a,b)
swap('int student_kernel_,student_decode_;','int student_kernel_,student_decode_,student_context_;uint32_t student_stamp_=0;std::vector<float> student_context_cache_;std::vector<uint32_t> student_context_marks_;')
swap('  void comp_context(double ratio,bool local,bool real){','''  void student_context_fill(int a,int full,int partial,float weight,const float* rp,bool local,bool real,float* out){
    alignas(32) float condition[8],vp[32];std::fill_n(vp,32,0.f);
    if(schema_&&local){for(int j=0;j<7;++j)condition[j]=real?view_code_[a*7+j]:0.f;condition[7]=1.f;view_.apply(condition,1,8,vp,32);}
    for(int o=0;o<32;o+=8){__m256 v=_mm256_loadu_ps(pprefix_+(int64_t(full)*A_+a)*32+o);if(partial>=0)v=_mm256_fmadd_ps(_mm256_set1_ps(weight),_mm256_loadu_ps(pfeature_+(int64_t(partial)*A_+a)*32+o),v);
      v=_mm256_add_ps(v,_mm256_load_ps(rp+o));v=_mm256_add_ps(v,_mm256_load_ps(vp+o));_mm256_storeu_ps(out+o,v);}
  }
  const float* student_dynamic(int a,int worker,int full,int partial,float weight,const float* rp,bool local,bool real){
    if(student_context_!=2)return comp_dynamic_.data()+a*32;
    int64_t slot=int64_t(worker)*A_+a;float* out=student_context_cache_.data()+slot*32;
    if(student_context_marks_[slot]!=student_stamp_){student_context_fill(a,full,partial,weight,rp,local,real,out);student_context_marks_[slot]=student_stamp_;}
    return out;
  }
  void student_context_begin(double ratio,bool local,bool real,int& full,int& partial,float& weight,float* rp){
    if(student_context_!=2){comp_context(ratio,local,real);return;}
    if(++student_stamp_==0){std::fill(student_context_marks_.begin(),student_context_marks_.end(),0);student_stamp_=1;}
    float code[3];controls(ratio,full,partial,weight,code);ratio_.apply(code,1,3,rp,32);
  }
  void comp_context(double ratio,bool local,bool real){''')
swap('    pool_.run(unique_anchors_.size(),64,[&](int begin,int end,int){','''    if(student_&&student_context_==1){
      pool_.run((unique_anchors_.size()+5)/6,16,[&](int begin,int end,int){alignas(32) float condition[6][8],vp[6][32];
        for(int block=begin;block<end;++block){int start=block*6,R=std::min(6,int(unique_anchors_.size())-start);std::fill_n(vp[0],6*32,0.f);
          if(schema_&&local){for(int r=0;r<R;++r){int a=unique_anchors_[start+r];for(int j=0;j<7;++j)condition[r][j]=real?view_code_[a*7+j]:0.f;condition[r][7]=1.f;}view_.apply(condition[0],R,8,vp[0],32);}
          for(int r=0;r<R;++r){int a=unique_anchors_[start+r];float* out=comp_dynamic_.data()+a*32;
            for(int o=0;o<32;o+=8){__m256 v=_mm256_loadu_ps(pprefix_+(int64_t(full)*A_+a)*32+o);if(partial>=0)v=_mm256_fmadd_ps(_mm256_set1_ps(weight),_mm256_loadu_ps(pfeature_+(int64_t(partial)*A_+a)*32+o),v);
              v=_mm256_add_ps(v,_mm256_load_ps(rp+o));v=_mm256_add_ps(v,_mm256_load_ps(vp[r]+o));_mm256_storeu_ps(out+o,v);}}
        }
      });return;
    }
    pool_.run(unique_anchors_.size(),64,[&](int begin,int end,int){''')
swap('int student_kernel=0,int student_decode=0)', 'int student_kernel=0,int student_decode=0,int student_context=0)')
swap('student_decode_(student_decode){','student_decode_(student_decode),student_context_(student_context){')
swap('    if(student_){TORCH_CHECK(H_==32,','''    TORCH_CHECK(student_context>=0&&student_context<=2&&(!student_context||student_kernel>=2),"Student context mode");
    if(student_&&student_context_==2){int64_t bytes=int64_t(threads)*A_*(32*sizeof(float)+sizeof(uint32_t));TORCH_CHECK(bytes<=256ll*1024*1024,"Student per-worker context exceeds256MiB");student_context_cache_.resize(int64_t(threads)*A_*32);student_context_marks_.assign(int64_t(threads)*A_,0);}
    if(student_){TORCH_CHECK(H_==32,''')
swap('    if(student_)bytes_+=student_body_plain_', '    bytes_+=student_context_cache_.size()*sizeof(float)+student_context_marks_.size()*sizeof(uint32_t);\n    if(student_)bytes_+=student_body_plain_')
swap('int,bool,bool,bool,bool,int,bool,int,int,int,int,int>()','int,bool,bool,bool,bool,int,bool,int,int,int,int,int,int>()')
swap('pybind11::arg("student_decode")=0)', 'pybind11::arg("student_decode")=0,pybind11::arg("student_context")=0)')
p.write_text(s,encoding='utf-8')
p=HERE/'source/student_fast.h';s=p.read_text(encoding='utf-8')
s=s.replace('  comp_context(ratio,local,real);','  int full=0,partial=-1;float weight=0;alignas(32) float rp[32];student_context_begin(ratio,local,real,full,partial,weight,rp);')
s=s.replace('int id=selected_[base+r],a=owners()[id];\n        for(int o=', 'int id=selected_[base+r],a=owners()[id];const float* dynamic=student_dynamic(a,worker,full,partial,weight,rp,local,real);\n        for(int o=')
s=s.replace('comp_dynamic_.data()+a*32+o','dynamic+o')
s=s.replace('int id=selected_[base+r],a=owners()[id];for(int j=0;j<32;++j)', 'int id=selected_[base+r],a=owners()[id];const float* dynamic=student_dynamic(a,worker,full,partial,weight,rp,local,real);for(int j=0;j<32;++j)').replace('comp_dynamic_[a*32+j]','dynamic[j]')
p.write_text(s,encoding='utf-8')
p=HERE/'kernel_runtime.py';s=p.read_text(encoding='utf-8').replace('student_kernel=0,student_decode=0):','student_kernel=0,student_decode=0,student_context=0):').replace('student_kernel,student_decode)','student_kernel,student_decode,student_context)');p.write_text(s,encoding='utf-8')
p=HERE/'common.py';s=p.read_text(encoding='utf-8').replace('control=(0,0),heads=(1,0),compact=(2,0),soa=(3,0),vector=(2,1),soa_vector=(3,1)','control=(0,0,0),heads=(1,0,0),compact=(2,0,0),soa=(3,0,0),vector=(2,1,0),soa_vector=(3,1,0),context=(2,1,1),lazy=(2,1,2)').replace('kernel,decode=modes[variant]','kernel,decode,context=modes[variant]').replace('student_decode=decode,**kwargs)','student_decode=decode,student_context=context,**kwargs)');p.write_text(s,encoding='utf-8')
print('CONTEXT_CANDIDATES_ADDED')
