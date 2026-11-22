// 专用2/3/4输出：保留原输入顺序和FMA链，去掉无消费者的上半AVX通道。
template<int Rows,int Inputs,int Outputs>
inline void apply_small(const float* x,int xs,float* y,int ys,bool activation)const{
  __m128 a0=_mm_setzero_ps(),a1=a0,a2=a0,a3=a0;
  if(small==2){
    __m128 b0=a0,b1=a0,b2=a0,b3=a0;
    #pragma GCC unroll 4
    for(int j=0;j<Inputs;j+=2){
      __m128 w0=_mm_loadu_ps(weights_ptr+j*8),w1=_mm_loadu_ps(weights_ptr+(j+1)*8);
      a0=_mm_fmadd_ps(_mm_set1_ps(x[j]),w0,a0);b0=_mm_fmadd_ps(_mm_set1_ps(x[j+1]),w1,b0);
      if constexpr(Rows>1){a1=_mm_fmadd_ps(_mm_set1_ps(x[xs+j]),w0,a1);b1=_mm_fmadd_ps(_mm_set1_ps(x[xs+j+1]),w1,b1);}
      if constexpr(Rows>2){a2=_mm_fmadd_ps(_mm_set1_ps(x[xs*2+j]),w0,a2);b2=_mm_fmadd_ps(_mm_set1_ps(x[xs*2+j+1]),w1,b2);}
      if constexpr(Rows>3){a3=_mm_fmadd_ps(_mm_set1_ps(x[xs*3+j]),w0,a3);b3=_mm_fmadd_ps(_mm_set1_ps(x[xs*3+j+1]),w1,b3);}
    }
    a0=_mm_add_ps(a0,b0);a1=_mm_add_ps(a1,b1);a2=_mm_add_ps(a2,b2);a3=_mm_add_ps(a3,b3);
  }else{
  #pragma GCC unroll 4
  for(int j=0;j<Inputs;++j){
    __m128 w=_mm_loadu_ps(weights_ptr+j*8);
    a0=_mm_fmadd_ps(_mm_set1_ps(x[j]),w,a0);
    if constexpr(Rows>1)a1=_mm_fmadd_ps(_mm_set1_ps(x[xs+j]),w,a1);
    if constexpr(Rows>2)a2=_mm_fmadd_ps(_mm_set1_ps(x[xs*2+j]),w,a2);
    if constexpr(Rows>3)a3=_mm_fmadd_ps(_mm_set1_ps(x[xs*3+j]),w,a3);
  }
  }
  __m128 values[4]={a0,a1,a2,a3},bias=_mm_loadu_ps(bias_ptr);
  #pragma GCC unroll 4
  for(int r=0;r<Rows;++r){__m128 v=_mm_add_ps(values[r],bias);
    if(activation)v=_mm_max_ps(_mm_setzero_ps(),v);
    alignas(16) float t[4];_mm_store_ps(t,v);std::memcpy(y+r*ys,t,Outputs*sizeof(float));
  }
}
template<int Rows,int Inputs>inline void small_outputs(const float* x,int xs,float* y,int ys,bool activation)const{
  if(out==2)apply_small<Rows,Inputs,2>(x,xs,y,ys,activation);
  else if(out==3)apply_small<Rows,Inputs,3>(x,xs,y,ys,activation);
  else apply_small<Rows,Inputs,4>(x,xs,y,ys,activation);
}
template<int Rows>inline void small_rows(const float* x,int xs,float* y,int ys,bool activation)const{
  if(in==32)small_outputs<Rows,32>(x,xs,y,ys,activation);else small_outputs<Rows,64>(x,xs,y,ys,activation);
}
