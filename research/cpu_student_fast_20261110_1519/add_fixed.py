"""固定Student32循环界及两次Newton旋转归一化，独立开关。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=HERE/'source/cpu_student_fast.cpp';s=p.read_text(encoding='utf-8')
start=s.index('template<int Rows,int Inputs>\ninline void apply_pair');end=s.index('template<int Rows> inline void fast_rows',start)
fixed=s[start:end].replace('template<int Rows,int Inputs>','template<int Rows,int Outputs>').replace('apply_pair(','apply_student32(').replace('const int inputs=Inputs?Inputs:in;','constexpr int inputs=32;').replace('o<out;o+=16','o<Outputs;o+=16')
s=s[:end]+fixed+s[end:]
s=s.replace('student_kernel<=3&&student_decode>=0&&student_decode<=1','student_kernel<=4&&student_decode>=0&&student_decode<=2')
p.write_text(s,encoding='utf-8')
p=HERE/'source/student_fast.h';s=p.read_text(encoding='utf-8')
prefix='''template<int Rows>void student_mlp32(const float* h0,float* h1,float* delta){
  comp1_.apply_student32<Rows,32>(h0,32,h1,32,true);student_head_.apply_student32<Rows,16>(h1,32,delta,16,false);
}
'''
s=prefix+s
old='      comp1_.apply(h0[0],R,32,h1[0],32,true);student_head_.apply(h1[0],R,32,delta[0],16);'
new='''      if(student_kernel_==4){
        switch(R){case 6:student_mlp32<6>(h0[0],h1[0],delta[0]);break;case 5:student_mlp32<5>(h0[0],h1[0],delta[0]);break;case 4:student_mlp32<4>(h0[0],h1[0],delta[0]);break;
          case 3:student_mlp32<3>(h0[0],h1[0],delta[0]);break;case 2:student_mlp32<2>(h0[0],h1[0],delta[0]);break;case 1:student_mlp32<1>(h0[0],h1[0],delta[0]);break;}
      }else{comp1_.apply(h0[0],R,32,h1[0],32,true);student_head_.apply(h1[0],R,32,delta[0],16);}'''
assert old in s;s=s.replace(old,new)
old='s=_mm256_max_ps(floor,_mm256_sqrt_ps(s));for(int j=0;j<4;++j)q[j]=_mm256_div_ps(q[j],s);'
new='''if(student_decode_==2){
      s=_mm256_max_ps(_mm256_set1_ps(1e-24f),s);__m256 r=_mm256_rsqrt_ps(s);
      for(int pass=0;pass<2;++pass){__m256 t=_mm256_mul_ps(_mm256_mul_ps(s,r),r);r=_mm256_mul_ps(r,_mm256_fnmadd_ps(_mm256_set1_ps(.5f),t,_mm256_set1_ps(1.5f)));}
      r=_mm256_blendv_ps(r,_mm256_setzero_ps(),_mm256_cmp_ps(s,_mm256_set1_ps(INFINITY),_CMP_EQ_OQ));for(int j=0;j<4;++j)q[j]=_mm256_mul_ps(q[j],r);
    }else{s=_mm256_max_ps(floor,_mm256_sqrt_ps(s));for(int j=0;j<4;++j)q[j]=_mm256_div_ps(q[j],s);}'''
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
p=HERE/'common.py';s=p.read_text(encoding='utf-8').replace('lazy=(2,1,2))','lazy=(2,1,2),fixed=(4,1,1),rsqrt=(2,2,1),fixed_rsqrt=(4,2,1))');p.write_text(s,encoding='utf-8')
print('FIXED32_RSQRT2_CANDIDATES_ADDED')
