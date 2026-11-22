"""V3：8角点同时投影，固定四链Linear微核与并行选择。"""
from pathlib import Path
import tarfile
HERE=Path(__file__).resolve().parent
with tarfile.open(HERE/'SOURCE_V2.tar.gz','x:gz') as archive:
    for n in ['source/cpu_native.cpp','cpu_runtime.py']:archive.add(HERE/n,arcname=n)
p=HERE/'source/cpu_native.cpp';s=p.read_text()
# 把4点accumulator显式固定，避免R动态导致寄存器spill和重复loop控制。
begin=s.index('  void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{')
end=s.index('\n};\n\nstruct Key',begin)
s=s[:begin]+'''  template<int Rows> inline void apply_rows(const float* x,int xs,float* y,int ys,bool activation)const{
    const float* weights=weights_ptr;const float* b=bias_ptr;
    for(int o=0;o<padded;o+=8){
      __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;const float* w=weights+(o/8)*in*8;
      for(int j=0;j<in;++j){__m256 v=_mm256_loadu_ps(w+j*8);
        a0=_mm256_fmadd_ps(_mm256_set1_ps(x[j]),v,a0);
        if constexpr(Rows>1)a1=_mm256_fmadd_ps(_mm256_set1_ps(x[xs+j]),v,a1);
        if constexpr(Rows>2)a2=_mm256_fmadd_ps(_mm256_set1_ps(x[xs*2+j]),v,a2);
        if constexpr(Rows>3)a3=_mm256_fmadd_ps(_mm256_set1_ps(x[xs*3+j]),v,a3);
      }
      __m256 result[4]={a0,a1,a2,a3};
      for(int r=0;r<Rows;++r){__m256 v=_mm256_add_ps(result[r],_mm256_loadu_ps(b+o));
        if(activation)v=_mm256_max_ps(_mm256_setzero_ps(),v);
        if(o+8<=out)_mm256_storeu_ps(y+r*ys+o,v);
        else{alignas(32) float tail[8];_mm256_store_ps(tail,v);for(int j=0;j<out-o;++j)y[r*ys+o+j]=tail[j];}}
    }
  }
  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{
    if(rows==4)apply_rows<4>(x,xs,y,ys,activation);else if(rows==3)apply_rows<3>(x,xs,y,ys,activation);
    else if(rows==2)apply_rows<2>(x,xs,y,ys,activation);else apply_rows<1>(x,xs,y,ys,activation);
  }
'''+s[end:]
start=s.index('    float lo[2]={INFINITY,INFINITY},hi[2]={-INFINITY,-INFINITY},minz=INFINITY,maxz=-INFINITY;')
end=s.index('    for(int j=0;j<7;++j)if(!std::isfinite(out[j]))',start)
s=s[:start]+'''    __m256 cx=_mm256_loadu_ps(corners_.data()+a*24),cy=_mm256_loadu_ps(corners_.data()+a*24+8),cz=_mm256_loadu_ps(corners_.data()+a*24+16);
    auto project=[&](const float* m,int c){__m256 v=_mm256_mul_ps(cx,_mm256_set1_ps(m[c]));v=_mm256_fmadd_ps(cy,_mm256_set1_ps(m[4+c]),v);v=_mm256_fmadd_ps(cz,_mm256_set1_ps(m[8+c]),v);return _mm256_add_ps(v,_mm256_set1_ps(m[12+c]));};
    auto reduce_min=[](__m256 v){__m128 q=_mm_min_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));q=_mm_min_ps(q,_mm_movehl_ps(q,q));q=_mm_min_ss(q,_mm_shuffle_ps(q,q,1));return _mm_cvtss_f32(q);};
    auto reduce_max=[](__m256 v){__m128 q=_mm_max_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));q=_mm_max_ps(q,_mm_movehl_ps(q,q));q=_mm_max_ss(q,_mm_shuffle_ps(q,q,1));return _mm_cvtss_f32(q);};
    __m256 depth=project(vm,2),cw=project(pm,3),absw=_mm256_andnot_ps(_mm256_set1_ps(-0.f),cw);
    __m256 nearv=_mm256_set1_ps(near),negative=_mm256_cmp_ps(cw,_mm256_setzero_ps(),_CMP_LT_OQ);
    __m256 signed_near=_mm256_blendv_ps(nearv,_mm256_sub_ps(_mm256_setzero_ps(),nearv),negative);
    cw=_mm256_blendv_ps(cw,signed_near,_mm256_cmp_ps(absw,nearv,_CMP_LT_OQ));
    __m256 sx=_mm256_div_ps(project(pm,0),cw),sy=_mm256_div_ps(project(pm,1),cw);
    float extent=clamp(std::max((reduce_max(sx)-reduce_min(sx))*aspect*.5f,(reduce_max(sy)-reduce_min(sy))*.5f),0,1e4f);
    if(reduce_min(depth)<=near)extent=1e4f;if(reduce_max(depth)<=near)extent=0;out[6]=std::log1p(extent);
'''+s[end:]
s=s.replace('corners_.resize(A_*32);','corners_.resize(A_*24);')
old='for(int c=0;c<8;++c){for(int j=0;j<3;++j)corners_[a*32+c*4+j]=f(0)[a*3+j]+f(1)[a]*.5f*((c&(1<<(2-j)))?1.f:-1.f);corners_[a*32+c*4+3]=1;}'
new='for(int c=0;c<8;++c)for(int j=0;j<3;++j)corners_[a*24+j*8+c]=f(0)[a*3+j]+f(1)[a]*.5f*((c&(1<<(2-j)))?1.f:-1.f);'
assert old in s;s=s.replace(old,new,1)
# 资格与分数直接产生复合键。数量不变，以稳定原行ID打破同分。
s=s.replace('keys_.reserve(N_);','keys_.resize(N_);')
old='      keys_.clear();for(int i=0;i<M_;++i)if(eligible()[rows_[i]])keys_.push_back({scores_[i],uint32_t(rows_[i])});'
new='''      keys_.resize(M_);pool_.run(M_,1024,[&](int begin,int end,int){for(int i=begin;i<end;++i)keys_[i]={eligible()[rows_[i]]?scores_[i]:-INFINITY,uint32_t(rows_[i])};});'''
assert old in s;s=s.replace(old,new,1)
# 全局百分位在全M键中选原rank，避免不合格负无穷产生错误。
s=s.replace('int index=int((1-ratio)*(M_-1)),rank=index-(M_-int(keys_.size()));float threshold=-INFINITY;',
    'int rank=int((1-ratio)*(M_-1));float threshold=-INFINITY;',1)
s=s.replace('for(auto& k:keys_)if(k.score>=threshold)selected_.push_back(k.id);','for(auto& k:keys_)if(eligible()[k.id]&&k.score>=threshold)selected_.push_back(k.id);',1)
s=s.replace('CpuPipelineV2','CpuPipelineV3');p.write_text(s)
p=HERE/'cpu_runtime.py';s=p.read_text().replace('build_v2','build_v3').replace('mycartgs_cpu_aggressive_v2','mycartgs_cpu_aggressive_v3');p.write_text(s)
print('CPU_V3_SIMD_VIEW_GENERATED')
