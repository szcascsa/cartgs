// 4点x16输出，保持每条FP32 FMA链及Bias/ReLU顺序。
template<int Rows,int Inputs>
inline void apply_pair(const float* x,int xs,float* y,int ys,bool activation)const{
  const int inputs=Inputs?Inputs:in;
  for(int o=0;o<out;o+=16){
    __m256 a00=_mm256_setzero_ps(),a01=a00,a10=a00,a11=a00,a20=a00,a21=a00,a30=a00,a31=a00;
    const float* w0=weights_ptr+(o/8)*inputs*8;const float* w1=w0+inputs*8;
    #pragma GCC unroll 4
    for(int j=0;j<inputs;++j){
      __m256 v0=_mm256_loadu_ps(w0+j*8),v1=_mm256_loadu_ps(w1+j*8),t=_mm256_set1_ps(x[j]);
      a00=_mm256_fmadd_ps(t,v0,a00);a01=_mm256_fmadd_ps(t,v1,a01);
      if constexpr(Rows>1){t=_mm256_set1_ps(x[xs+j]);a10=_mm256_fmadd_ps(t,v0,a10);a11=_mm256_fmadd_ps(t,v1,a11);}
      if constexpr(Rows>2){t=_mm256_set1_ps(x[xs*2+j]);a20=_mm256_fmadd_ps(t,v0,a20);a21=_mm256_fmadd_ps(t,v1,a21);}
      if constexpr(Rows>3){t=_mm256_set1_ps(x[xs*3+j]);a30=_mm256_fmadd_ps(t,v0,a30);a31=_mm256_fmadd_ps(t,v1,a31);}
    }
    __m256 lo[4]={a00,a10,a20,a30},hi[4]={a01,a11,a21,a31};
    __m256 b0=_mm256_loadu_ps(bias_ptr+o),b1=_mm256_loadu_ps(bias_ptr+o+8);
    #pragma GCC unroll 4
    for(int r=0;r<Rows;++r){
      __m256 v0=_mm256_add_ps(lo[r],b0),v1=_mm256_add_ps(hi[r],b1);
      if(activation){v0=_mm256_max_ps(_mm256_setzero_ps(),v0);v1=_mm256_max_ps(_mm256_setzero_ps(),v1);}
      _mm256_storeu_ps(y+r*ys+o,v0);_mm256_storeu_ps(y+r*ys+o+8,v1);
    }
  }
}
template<int Rows> inline void fast_rows(const float* x,int xs,float* y,int ys,bool activation)const{
  if(in==32)apply_pair<Rows,32>(x,xs,y,ys,activation);
  else if(in==64)apply_pair<Rows,64>(x,xs,y,ys,activation);
  else apply_pair<Rows,0>(x,xs,y,ys,activation);
}
