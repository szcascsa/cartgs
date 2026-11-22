"""仅独立研究目录生成Student专用路径；历史后端保持不变。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;KERNEL=HERE.parent/'cpu_native_midratio_20261110_1407';PREVIOUS=HERE.parent/'cpu_student_inference_20261110_1448'
s=(KERNEL/'source/cpu_mid.cpp').read_text(encoding='utf-8').replace('CpuPipelineMid','CpuPipelineStudentFast')
def swap(a,b):
    global s
    assert s.count(a)==1,(a[:80],s.count(a));s=s.replace(a,b)
swap('using T=at::Tensor;', 'extern "C" __m256 _ZGVdN8v_expf(__m256);\nextern "C" __m256 _ZGVdN8v_logf(__m256);\nusing T=at::Tensor;')
swap('Dense combined_head_;','Dense combined_head_,student_head_;T student_body_plain_,student_head_plain_;int student_kernel_,student_decode_;')
swap('  void compensate(double ratio,bool local,bool real,float* output,bool enabled){', '#include "student_fast.h"\n  void compensate(double ratio,bool local,bool real,float* output,bool enabled){')
swap('    comp_context(ratio,local,real);\n    int full,partial;', '    if(student_&&factored_&&student_kernel_>=2){if(student_kernel_==3)student_soa(ratio,local,real,output);else student_compact(ratio,local,real,output);return;}\n    comp_context(ratio,local,real);\n    int full,partial;')
swap('      alignas(32) float h0[6][64],h1[6][64],hidden[6][256],small[6][4],delta[6][14],input[6][56],rp[64],vp[64];', '      alignas(32) float h0[6][64],h1[6][64],hidden[6][256],small[6][4],packed[6][16],delta[6][14],input[6][56],rp[64],vp[64];')
start=s.index('        for(int head=0;head<4;++head){const float* source=student_?h1[0]');end=s.index('        for(int r=0;r<R;++r){int id=selected_[base+r];float reference=',start)
old=s[start:end];new='''        if(student_&&student_kernel_){student_head_.apply(h1[0],R,64,packed[0],16);for(int r=0;r<R;++r)std::copy_n(packed[r],14,delta[r]);}
        else{
'''+old+'''        }
''';s=s[:start]+new+s[end:]
swap('int selection_mode=0,int comp_batch=4)', 'int selection_mode=0,int comp_batch=4,int student_kernel=0,int student_decode=0)')
swap('comp_batch_(comp_batch),pool_(threads)', 'comp_batch_(comp_batch),pool_(threads)') # initializer order follows member declaration below.
swap('schema_(schema),scene_scale_(scene_scale){', 'schema_(schema),scene_scale_(scene_scale),student_kernel_(student_kernel),student_decode_(student_decode){')
needle='    TORCH_CHECK(kernel_mode>=0&&kernel_mode<=1,"Unknown AVX2 kernel mode");'
swap(needle,'''    TORCH_CHECK(student_kernel>=0&&student_kernel<=3&&student_decode>=0&&student_decode<=1&&(!student_decode||student_kernel>=2),"Student optimization modes");
    if(student_){TORCH_CHECK(H_==32,"Student fast path requires32D");auto weight=at::zeros({16,32},comp[2].options()),bias=at::zeros({16},comp[3].options());int offset=0;
      for(int head=0;head<4;++head){int n=head<2?3:4;weight.narrow(0,offset,n).copy_(comp[5+head*2]);bias.narrow(0,offset,n).copy_(comp[6+head*2]);offset+=n;}
      student_head_=Dense(weight,bias);student_head_.fast=1;student_body_plain_=own(comp[2]);student_head_plain_=own(weight);}
'''+needle)
swap('    bytes_+=int64_t(N_)*(sizeof(int64_t)*2+sizeof(Key))', '    if(student_)bytes_+=student_body_plain_.nbytes()+student_head_plain_.nbytes()+student_head_.packed.nbytes()+student_head_.bias.nbytes();\n    bytes_+=int64_t(N_)*(sizeof(int64_t)*2+sizeof(Key))')
swap('int,bool,bool,bool,bool,int,bool,int,int,int>()','int,bool,bool,bool,bool,int,bool,int,int,int,int,int>()')
swap('pybind11::arg("comp_batch")=4)', 'pybind11::arg("comp_batch")=4,pybind11::arg("student_kernel")=0,pybind11::arg("student_decode")=0)')
(HERE/'source/cpu_student_fast.cpp').write_text(s,encoding='utf-8')
(HERE/'source/native_handoff.h').write_text((KERNEL/'source/native_handoff.h').read_text(encoding='utf-8').replace('CpuPipelineMid','CpuPipelineStudentFast').replace('NativeHandoffMid','NativeHandoffStudentFast').replace('NativeTicketMid','NativeTicketStudentFast'),encoding='utf-8')
r=(KERNEL/'mid_runtime.py').read_text(encoding='utf-8').replace('mycartgs_cpu_mid','mycartgs_cpu_student_fast').replace('cpu_mid.cpp','cpu_student_fast.cpp')
r=r.replace('selection_mode=1,comp_batch=6):','selection_mode=1,comp_batch=6,student_kernel=0,student_decode=0):').replace('selection_mode,comp_batch)','selection_mode,comp_batch,student_kernel,student_decode)')
r=r.replace("'-lcudart','-lc10_cuda','-ltorch_cuda'","'-lcudart','-lc10_cuda','-ltorch_cuda','-lmvec','-lm'")
(HERE/'kernel_runtime.py').write_text(r,encoding='utf-8')
r=(PREVIOUS/'student_runtime.py').read_text(encoding='utf-8').replace("BASE=HERE.parent/'cpu_native_midratio_20261110_1407'","BASE=HERE").replace('from mid_runtime import CpuRenderer as TeacherCpuRenderer,extension','from kernel_runtime import CpuRenderer as TeacherCpuRenderer,extension')
(HERE/'student_fast_runtime.py').write_text(r,encoding='utf-8');print('STUDENT_FAST_GENERATED',len(s))
