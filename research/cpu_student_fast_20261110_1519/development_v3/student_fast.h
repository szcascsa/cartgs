template<int Rows>void student_mlp32(const float* h0,float* h1,float* delta){
  comp1_.apply_student32<Rows,32>(h0,32,h1,32,true);student_head_.apply_student32<Rows,16>(h1,32,delta,16,false);
}
// 仅冻结32/32学生。权重来自原四个独立Head；16列含两列固定padding。
void student_compact(double ratio,bool local,bool real,float* output){
  int full=0,partial=-1;float weight=0;alignas(32) float rp[32];student_context_begin(ratio,local,real,full,partial,weight,rp);
  pool_.run((K_+5)/6,16,[&](int begin,int end,int worker){
    alignas(32) float h0[8][32],h1[8][32],delta[8][16]{};
    for(int block=begin;block<end;++block){int base=block*6,R=std::min(6,K_-base);
      for(int r=0;r<R;++r){int id=selected_[base+r],a=owners()[id];const float* dynamic=student_dynamic(a,worker,full,partial,weight,rp,local,real);
        for(int o=0;o<32;o+=8){__m256 v=_mm256_add_ps(_mm256_loadu_ps(poffset_+id*32+o),_mm256_loadu_ps(dynamic+o));_mm256_store_ps(h0[r]+o,_mm256_max_ps(_mm256_setzero_ps(),v));}}
      if(student_kernel_==4){
        switch(R){case 6:student_mlp32<6>(h0[0],h1[0],delta[0]);break;case 5:student_mlp32<5>(h0[0],h1[0],delta[0]);break;case 4:student_mlp32<4>(h0[0],h1[0],delta[0]);break;
          case 3:student_mlp32<3>(h0[0],h1[0],delta[0]);break;case 2:student_mlp32<2>(h0[0],h1[0],delta[0]);break;case 1:student_mlp32<1>(h0[0],h1[0],delta[0]);break;}
      }else{comp1_.apply(h0[0],R,32,h1[0],32,true);student_head_.apply(h1[0],R,32,delta[0],16);}
      for(int r=0;r<R;++r){int id=selected_[base+r];float reference=f(1)[owners()[id]];
        for(int j=0;j<3;++j)delta[r][j]*=reference;
        for(int j=0;j<14;++j)if(!std::isfinite(delta[r][j]))bad_.store(3,std::memory_order_relaxed);
        if(diagnose_)std::copy_n(delta[r],14,residual_.data()+(base+r)*14);}
      if(student_decode_)student_publish_vector(base,R,delta[0],output);
      else for(int r=0;r<R;++r)publish(base+r,selected_[base+r],delta[r],output,true);
    }
  });
}

// 8点SoA微核：每lane一个Gaussian，权重标量广播，保留每输出的j顺序。
void student_soa(double ratio,bool local,bool real,float* output){
  int full=0,partial=-1;float weight=0;alignas(32) float rp[32];student_context_begin(ratio,local,real,full,partial,weight,rp);
  pool_.run((K_+7)/8,16,[&](int begin,int end,int worker){
    alignas(32) float h0[32][8]{},h1[32][8],delta[8][16]{};
    const float* weights=student_body_plain_.data_ptr<float>();const float* heads=student_head_plain_.data_ptr<float>();
    for(int block=begin;block<end;++block){int base=block*8,R=std::min(8,K_-base);
      for(int r=0;r<R;++r){int id=selected_[base+r],a=owners()[id];const float* dynamic=student_dynamic(a,worker,full,partial,weight,rp,local,real);for(int j=0;j<32;++j){float v=poffset_[id*32+j]+dynamic[j];h0[j][r]=v<0.f?0.f:v;}}
      for(int o=0;o<32;o+=8){__m256 v[8];for(auto& a:v)a=_mm256_setzero_ps();
        for(int j=0;j<32;++j){__m256 x=_mm256_load_ps(h0[j]);
          #pragma GCC unroll 8
          for(int t=0;t<8;++t)v[t]=_mm256_fmadd_ps(x,_mm256_set1_ps(weights[(o+t)*32+j]),v[t]);}
        for(int t=0;t<8;++t)_mm256_store_ps(h1[o+t],_mm256_max_ps(_mm256_setzero_ps(),_mm256_add_ps(v[t],_mm256_set1_ps(comp1_.bias_ptr[o+t]))));}
      for(int o=0;o<16;o+=8){__m256 v[8];for(auto& a:v)a=_mm256_setzero_ps();
        for(int j=0;j<32;++j){__m256 x=_mm256_load_ps(h1[j]);
          #pragma GCC unroll 8
          for(int t=0;t<8;++t)v[t]=_mm256_fmadd_ps(x,_mm256_set1_ps(heads[(o+t)*32+j]),v[t]);}
        for(int t=0;t<8;++t){alignas(32) float values[8];_mm256_store_ps(values,_mm256_add_ps(v[t],_mm256_set1_ps(student_head_.bias_ptr[o+t])));for(int r=0;r<R;++r)delta[r][o+t]=values[r];}}
      for(int r=0;r<R;++r){int id=selected_[base+r];float reference=f(1)[owners()[id]];for(int j=0;j<3;++j)delta[r][j]*=reference;
        for(int j=0;j<14;++j)if(!std::isfinite(delta[r][j]))bad_.store(3,std::memory_order_relaxed);
        if(diagnose_)std::copy_n(delta[r],14,residual_.data()+(base+r)*14);}
      if(student_decode_)student_publish_vector(base,R,delta[0],output);
      else for(int r=0;r<R;++r)publish(base+r,selected_[base+r],delta[r],output,true);
    }
  });
}

void student_publish_vector(int base,int R,const float* deltas,float* output){
  alignas(32) int ids[8];for(int r=0;r<8;++r)ids[r]=selected_[base+std::min(r,R-1)];
  __m256i index=_mm256_load_si256(reinterpret_cast<const __m256i*>(ids)),i3=_mm256_mullo_epi32(index,_mm256_set1_epi32(3)),i4=_mm256_slli_epi32(index,2);
  __m256i rows=_mm256_setr_epi32(0,16,32,48,64,80,96,112);auto delta=[&](int j){return _mm256_i32gather_ps(deltas+j,rows,4);};
  auto three=[&](const float* p,int j){return _mm256_i32gather_ps(p+j,i3,4);};
  const __m256 zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1.f),floor=_mm256_set1_ps(1e-12f);
  auto norm4=[&](__m256* q){__m256 s=_mm256_mul_ps(q[0],q[0]);for(int j=1;j<4;++j)s=_mm256_add_ps(s,_mm256_mul_ps(q[j],q[j]));if(student_decode_==2){
      s=_mm256_max_ps(_mm256_set1_ps(1e-24f),s);__m256 r=_mm256_rsqrt_ps(s);
      for(int pass=0;pass<2;++pass){__m256 t=_mm256_mul_ps(_mm256_mul_ps(s,r),r);r=_mm256_mul_ps(r,_mm256_fnmadd_ps(_mm256_set1_ps(.5f),t,_mm256_set1_ps(1.5f)));}
      r=_mm256_blendv_ps(r,_mm256_setzero_ps(),_mm256_cmp_ps(s,_mm256_set1_ps(INFINITY),_CMP_EQ_OQ));for(int j=0;j<4;++j)q[j]=_mm256_mul_ps(q[j],r);
    }else{s=_mm256_max_ps(floor,_mm256_sqrt_ps(s));for(int j=0;j<4;++j)q[j]=_mm256_div_ps(q[j],s);}};
  __m256 d[4]={_mm256_add_ps(one,delta(6)),delta(7),delta(8),delta(9)},q[4];norm4(d);
  for(int j=0;j<4;++j)q[j]=_mm256_i32gather_ps(f(6)+j,i4,4);
  __m256 v[4]={
    _mm256_sub_ps(_mm256_mul_ps(d[0],q[0]),_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(d[1],q[1]),_mm256_mul_ps(d[2],q[2])),_mm256_mul_ps(d[3],q[3]))),
    _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(d[0],q[1]),_mm256_mul_ps(q[0],d[1])),_mm256_sub_ps(_mm256_mul_ps(d[2],q[3]),_mm256_mul_ps(d[3],q[2]))),
    _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(d[0],q[2]),_mm256_mul_ps(q[0],d[2])),_mm256_sub_ps(_mm256_mul_ps(d[3],q[1]),_mm256_mul_ps(d[1],q[3]))),
    _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(d[0],q[3]),_mm256_mul_ps(q[0],d[3])),_mm256_sub_ps(_mm256_mul_ps(d[1],q[2]),_mm256_mul_ps(d[2],q[1])))};
  norm4(v);norm4(v);alignas(32) float values[14][8];
  for(int j=0;j<4;++j)_mm256_store_ps(values[7+j],v[j]);
  __m256 valid=_mm256_castsi256_ps(_mm256_set1_epi32(-1));const __m256 infinity=_mm256_set1_ps(INFINITY),sign=_mm256_set1_ps(-0.f);
  auto finite=[&](__m256 x){valid=_mm256_and_ps(valid,_mm256_cmp_ps(_mm256_andnot_ps(sign,x),infinity,_CMP_LT_OQ));};
  for(int j=0;j<3;++j){__m256 xyz=_mm256_add_ps(three(f(4),j),delta(j)),scale=three(activated_scale_.data(),j);
    if(softplus_){__m256 x=_mm256_add_ps(delta(3+j),_mm256_set1_ps(.5413248546129181f));
      __m256 e=_ZGVdN8v_expf(_mm256_min_ps(x,_mm256_set1_ps(20.f))),y=_mm256_add_ps(one,e);
      // log1p(e)=log(y)+(e-(y-1))/y，保留小e，不用log(1+e)直接抹掉尾部。
      __m256 log1p=_mm256_add_ps(_ZGVdN8v_logf(y),_mm256_div_ps(_mm256_sub_ps(e,_mm256_sub_ps(y,one)),y));
      __m256 soft=_mm256_blendv_ps(log1p,x,_mm256_cmp_ps(x,_mm256_set1_ps(20.f),_CMP_GT_OQ));
      scale=_mm256_max_ps(_mm256_set1_ps(std::numeric_limits<float>::min()),_mm256_mul_ps(scale,soft));
    }else scale=_mm256_add_ps(scale,delta(3+j));
    __m256 dc=_mm256_add_ps(three(f(8),j),delta(11+j));_mm256_store_ps(values[j],xyz);_mm256_store_ps(values[4+j],scale);_mm256_store_ps(values[11+j],dc);
    finite(xyz);finite(scale);finite(dc);valid=_mm256_and_ps(valid,_mm256_cmp_ps(scale,zero,_CMP_GT_OQ));}
  __m256 logit=_mm256_add_ps(_mm256_i32gather_ps(f(7),index,4),delta(10));__m256 opacity=_mm256_div_ps(one,_mm256_add_ps(one,_ZGVdN8v_expf(_mm256_sub_ps(zero,logit))));
  _mm256_store_ps(values[3],opacity);finite(opacity);for(auto x:v)finite(x);if((_mm256_movemask_ps(valid)&((1<<R)-1))!=((1<<R)-1))bad_.store(3,std::memory_order_relaxed);
  int64_t k=K_;float* xyz=output;float* op=xyz+k*3;float* scales=op+k;float* rotations=scales+k*3;float* dc=rotations+k*4;float* rest=dc+k*3;float* means=rest+k*rest_;
  for(int r=0;r<R;++r){int row=base+r;for(int j=0;j<3;++j){xyz[row*3+j]=values[j][r];scales[row*3+j]=values[4+j][r];dc[row*3+j]=values[11+j][r];means[row*3+j]=0;}
    op[row]=values[3][r];for(int j=0;j<4;++j)rotations[row*4+j]=values[7+j][r];std::copy_n(f(9)+ids[r]*rest_,rest_,rest+row*rest_);}
}
