// 独立冻结CPU部署：FP32 AVX2/FMA，允许已测的小舍入差异，不用于训练。
#include <torch/extension.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

using T=at::Tensor;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
static T own(const T& x,at::ScalarType type=at::kFloat){
  TORCH_CHECK(x.device().is_cpu()&&x.scalar_type()==type,"Expected CPU tensor with matching dtype");
  auto y=x.detach().contiguous().clone();
  if(type==at::kFloat)TORCH_CHECK(at::isfinite(y).all().item<bool>(),"Nonfinite frozen input");
  return y;
}

class Pool {
  int count_;std::vector<std::thread> workers_;std::mutex mutex_;std::condition_variable ready_,done_;
  std::atomic<uint64_t> generation_{0};std::atomic<int> remaining_{0};std::atomic<bool> stopping_{false};
  std::function<void(int,int,int)> function_;int size_=0;
  void part(int id){int begin=int(int64_t(size_)*id/count_),end=int(int64_t(size_)*(id+1)/count_);function_(begin,end,id);}
  void worker(int id){
    uint64_t seen=0;
    while(true){
      int spins=0;
      while(generation_.load(std::memory_order_acquire)==seen&&!stopping_.load(std::memory_order_relaxed)&&spins++<12000)_mm_pause();
      if(generation_.load(std::memory_order_acquire)==seen&&!stopping_.load()){
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock,[&]{return stopping_.load()||generation_.load(std::memory_order_acquire)!=seen;});
      }
      if(stopping_.load())return;
      seen=generation_.load(std::memory_order_acquire);part(id);
      if(remaining_.fetch_sub(1,std::memory_order_acq_rel)==1){std::lock_guard<std::mutex> lock(mutex_);done_.notify_one();}
    }
  }
public:
  explicit Pool(int count):count_(count){TORCH_CHECK(count>=1&&count<=64,"CPU worker count must be 1..64");for(int i=1;i<count;++i)workers_.emplace_back([this,i]{worker(i);});}
  ~Pool(){stopping_=true;{std::lock_guard<std::mutex> lock(mutex_);ready_.notify_all();}for(auto& w:workers_)w.join();}
  int count()const{return count_;}
  void run(int n,int minimum,const std::function<void(int,int,int)>& f){
    if(count_==1||n<minimum){f(0,n,0);return;}
    function_=f;size_=n;remaining_.store(count_-1,std::memory_order_relaxed);
    {std::lock_guard<std::mutex> lock(mutex_);generation_.fetch_add(1,std::memory_order_release);ready_.notify_all();}
    part(0);int spins=0;while(remaining_.load(std::memory_order_acquire)&&spins++<12000)_mm_pause();
    if(remaining_.load(std::memory_order_acquire)){std::unique_lock<std::mutex> lock(mutex_);done_.wait(lock,[&]{return remaining_.load(std::memory_order_acquire)==0;});}
  }
};

// 权重以8个输出神经元为连续块；同一权重向量供4条独立点链复用。
struct Dense {
  int in=0,out=0,padded=0;bool fast=false;T packed,bias;const float* weights_ptr=nullptr;const float* bias_ptr=nullptr;
  Dense()=default;
  Dense(const T& w,const T& b){
    auto v=own(w);auto q=own(b);TORCH_CHECK(v.dim()==2&&q.numel()==v.size(0),"Linear weight shape");
    out=v.size(0);in=v.size(1);padded=(out+7)/8*8;
    packed=at::zeros({padded*in},v.options());bias=at::zeros({padded},v.options());
    auto src=v.data_ptr<float>();auto dst=packed.data_ptr<float>();
    for(int o=0;o<out;++o){bias.data_ptr<float>()[o]=q.data_ptr<float>()[o];for(int j=0;j<in;++j)dst[(o/8)*in*8+j*8+o%8]=src[o*in+j];}
    weights_ptr=packed.data_ptr<float>();bias_ptr=bias.data_ptr<float>();
  }
  template<int Rows> inline void apply_rows(const float* x,int xs,float* y,int ys,bool activation)const{
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

  inline void apply(const float* x,int rows,int xs,float* y,int ys,bool activation=false)const{
    if(fast&&out%16==0){
      if(rows==4)fast_rows<4>(x,xs,y,ys,activation);else if(rows==3)fast_rows<3>(x,xs,y,ys,activation);
      else if(rows==2)fast_rows<2>(x,xs,y,ys,activation);else fast_rows<1>(x,xs,y,ys,activation);return;
    }
    if(rows==4)apply_rows<4>(x,xs,y,ys,activation);else if(rows==3)apply_rows<3>(x,xs,y,ys,activation);
    else if(rows==2)apply_rows<2>(x,xs,y,ys,activation);else apply_rows<1>(x,xs,y,ys,activation);
  }

};

struct Key {float score;uint32_t id;};
struct Better {bool operator()(const Key& a,const Key& b)const{return a.score>b.score||(a.score==b.score&&a.id<b.id);}};

class CpuPipelineDeep {
  bool fused_schedule_;int kernel_mode_;
  Pool pool_;std::mutex call_;std::vector<T> s_,tables_;std::vector<double> knots_,bounds_;
  Dense sel0_,sel1_,sel2_,comp0_,comp1_,ratio_,view_;
  std::vector<Dense> hh_,ho_;Dense combined_head_;std::vector<float> activated_scale_,activated_opacity_;bool student_,softplus_,selector_prefix_,factored_,diagnose_;
  int N_,A_,H_,schema_,rest_,width_,M_=0,K_=0;float scene_scale_;
  T selector_prefix_data_,offset_prefix_,feature_projected_,feature_prefix_;
  std::vector<float> centers_soa_,corners_,view_code_,sel_dynamic_,comp_dynamic_,scores_,residual_;
  std::vector<uint8_t> visible_,occupied_;std::vector<int64_t> rows_,selected_;std::vector<Key> keys_;
  std::vector<int> counts_,eligible_counts_,offsets_,unique_anchors_;std::vector<uint32_t> marks_;uint32_t stamp_=0;
  std::atomic<int> bad_{0};int64_t bytes_=0;
  const float* F_[13]{};const int64_t* owner_ptr_=nullptr;const bool* eligible_ptr_=nullptr;
  float *pselector_=nullptr,*poffset_=nullptr,*pfeature_=nullptr,*pprefix_=nullptr;std::vector<const float*> table_ptrs_;
  const float* f(int i)const{return F_[i];}
  const int64_t* owners()const{return owner_ptr_;}
  const bool* eligible()const{return eligible_ptr_;}
  static float dot4(const float* p,const float* m,int c){float v=0;for(int j=0;j<4;++j)v=std::fma(p[j],m[j*4+c],v);return v;}
  static float safe(float w,float near){return std::fabs(w)<near?(w<0?-near:near):w;}
  static float clamp(float v,float lo,float hi){return std::min(hi,std::max(lo,v));}
  void encode_view(int a,const float* vm,const float* pm,const float* camera,float aspect,float near){
    const float* center=f(0)+a*3;float* out=view_code_.data()+a*7;float q[4]={center[0],center[1],center[2],1.f};
    float v[3]={camera[0]-center[0],camera[1]-center[1],camera[2]-center[2]};
    float norm=std::max(std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]),1e-6f);
    for(int j=0;j<3;++j)out[j]=v[j]/norm;
    out[3]=std::asinh(dot4(q,vm,2)/scene_scale_);float w=safe(dot4(q,pm,3),near);
    out[4]=clamp(dot4(q,pm,0)/w,-4,4);out[5]=clamp(dot4(q,pm,1)/w,-4,4);
    __m256 cx=_mm256_loadu_ps(corners_.data()+a*24),cy=_mm256_loadu_ps(corners_.data()+a*24+8),cz=_mm256_loadu_ps(corners_.data()+a*24+16);
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
    for(int j=0;j<7;++j)if(!std::isfinite(out[j]))bad_.store(1,std::memory_order_relaxed);
  }
  void anchor_context(int a,const float* vm,const float* pm,const float* camera,float aspect,float near,bool local,bool compensation){
    bool selector_real=schema_==2||schema_==3,comp_real=schema_==2||schema_==4;
    if(local&&schema_&&(selector_real||(compensation&&comp_real)))encode_view(a,vm,pm,camera,aspect,near);
    if(schema_){alignas(32) float x[8];for(int j=0;j<7;++j)x[j]=local&&selector_real?view_code_[a*7+j]:0.f;x[7]=local?1.f:0.f;
      for(int o=0;o<32;o+=8){__m256 v=_mm256_setzero_ps();for(int j=0;j<8;++j)v=_mm256_fmadd_ps(_mm256_set1_ps(x[j]),_mm256_loadu_ps(sel0_.weights_ptr+(o/8)*sel0_.in*8+(j+11)*8),v);_mm256_storeu_ps(sel_dynamic_.data()+a*32+o,v);}}
  }
  void frustum(const float* pm,bool global,const float* vm,const float* camera,float aspect,float near,bool local,bool compensation,bool context){
    float planes[6][4];for(int j=0;j<4;++j){planes[0][j]=pm[j*4+3]+pm[j*4];planes[1][j]=pm[j*4+3]-pm[j*4];planes[2][j]=pm[j*4+3]+pm[j*4+1];planes[3][j]=pm[j*4+3]-pm[j*4+1];planes[4][j]=pm[j*4+2];planes[5][j]=pm[j*4+3]-pm[j*4+2];}
    pool_.run((A_+7)/8,32,[&](int begin,int end,int){
      for(int block=begin;block<end;++block){int a=block*8;
        if(global){for(int j=a;j<std::min(a+8,A_);++j){visible_[j]=occupied_[j];if(context&&visible_[j])anchor_context(j,vm,pm,camera,aspect,near,local,compensation);}continue;}
        if(a+8<=A_){
          __m256 x=_mm256_loadu_ps(centers_soa_.data()+a),y=_mm256_loadu_ps(centers_soa_.data()+A_+a),z=_mm256_loadu_ps(centers_soa_.data()+2*A_+a),h=_mm256_loadu_ps(centers_soa_.data()+3*A_+a);
          __m256 keep=_mm256_castsi256_ps(_mm256_set1_epi32(-1));
          for(int p=0;p<6;++p){float* v=planes[p];__m256 d=_mm256_mul_ps(x,_mm256_set1_ps(v[0]));d=_mm256_fmadd_ps(y,_mm256_set1_ps(v[1]),d);d=_mm256_fmadd_ps(z,_mm256_set1_ps(v[2]),d);d=_mm256_add_ps(d,_mm256_set1_ps(v[3]));
            d=_mm256_add_ps(d,_mm256_mul_ps(h,_mm256_set1_ps(std::fabs(v[0])+std::fabs(v[1])+std::fabs(v[2]))));keep=_mm256_and_ps(keep,_mm256_cmp_ps(d,_mm256_setzero_ps(),_CMP_GE_OQ));}
          unsigned mask=_mm256_movemask_ps(keep);for(int j=0;j<8;++j)visible_[a+j]=bool(mask&(1u<<j))&&occupied_[a+j];
        }else for(;a<A_;++a){bool keep=occupied_[a];for(auto& p:planes){float d=centers_soa_[a]*p[0]+centers_soa_[A_+a]*p[1]+centers_soa_[A_*2+a]*p[2]+p[3];d+=centers_soa_[A_*3+a]*(std::fabs(p[0])+std::fabs(p[1])+std::fabs(p[2]));keep=keep&&d>=0;}visible_[a]=keep;}
        if(context)for(int j=block*8;j<std::min(block*8+8,A_);++j)if(visible_[j])anchor_context(j,vm,pm,camera,aspect,near,local,compensation);
      }
    });
  }
  void candidates(bool global){
    if(global){M_=N_;for(int i=0;i<N_;++i)rows_[i]=i;return;}
    std::fill(counts_.begin(),counts_.end(),0);
    pool_.run(N_,0,[&](int begin,int end,int thread){int count=0;for(int i=begin;i<end;++i)count+=visible_[owners()[i]];counts_[thread]=count;});
    M_=0;for(int i=0;i<pool_.count();++i){offsets_[i]=M_;M_+=counts_[i];}
    pool_.run(N_,0,[&](int begin,int end,int thread){int out=offsets_[thread];for(int i=begin;i<end;++i)if(visible_[owners()[i]])rows_[out++]=i;});
  }
  void selector(float log_ratio,bool local,bool real_view){
    pool_.run((M_+3)/4,16,[&](int begin,int end,int){
      alignas(32) float x[4][24],h0[4][32],h1[4][32],logits[4][2];
      const float* w=sel0_.weights_ptr;const float* bias=sel0_.bias_ptr;
      for(int block=begin;block<end;++block){int base=block*4,R=std::min(4,M_-base);
        if(selector_prefix_){
          for(int r=0;r<R;++r){int id=rows_[base+r],a=owners()[id];const float* pre=pselector_+id*32;
            for(int o=0;o<32;o+=8){__m256 v=_mm256_add_ps(_mm256_loadu_ps(pre+o),_mm256_mul_ps(_mm256_set1_ps(log_ratio),_mm256_loadu_ps(w+(o/8)*sel0_.in*8+10*8)));
              v=_mm256_add_ps(v,_mm256_loadu_ps(bias+o));if(schema_)v=_mm256_add_ps(v,_mm256_loadu_ps(sel_dynamic_.data()+a*32+o));
              _mm256_store_ps(h0[r]+o,_mm256_max_ps(_mm256_setzero_ps(),v));}}
        }else{
          for(int r=0;r<R;++r){int id=rows_[base+r],a=owners()[id];std::copy_n(f(10)+id*10,10,x[r]);x[r][10]=log_ratio;
            if(schema_){for(int j=0;j<7;++j)x[r][j+11]=local&&real_view?view_code_[a*7+j]:0.f;x[r][18]=local?1.f:0.f;}}
          sel0_.apply(x[0],R,24,h0[0],32,true);
        }
        sel1_.apply(h0[0],R,32,h1[0],32,true);sel2_.apply(h1[0],R,32,logits[0],2);
        for(int r=0;r<R;++r){int id=rows_[base+r];float a=logits[r][0]+f(12)[id*2],b=logits[r][1]+f(12)[id*2+1],max=std::max(a,b);float ea=std::exp(a-max),eb=std::exp(b-max);scores_[base+r]=eb/(ea+eb);if(fused_schedule_)keys_[base+r]={eligible()[id]?scores_[base+r]:-INFINITY,uint32_t(id)};
          if(!std::isfinite(logits[r][0])||!std::isfinite(logits[r][1])||!std::isfinite(scores_[base+r]))bad_.store(2,std::memory_order_relaxed);}
      }
    });
  }
  void controls(double ratio,int& full,int& partial,float& weight,float* code){
    full=0;partial=-1;weight=0;
    for(int j=1;j<int(knots_.size());++j){if(ratio<=knots_[j-1]+1e-8)break;double t=std::min(1.,(ratio-knots_[j-1])/(knots_[j]-knots_[j-1]));if(t==1.)full=j;else{partial=j;weight=float(t);break;}}
    double t=2*(std::log(ratio)-std::log(bounds_[0]))/(std::log(bounds_[1])-std::log(bounds_[0]))-1;
    code[0]=t;code[1]=std::sin(3.14159265358979323846*t/2);code[2]=std::cos(3.14159265358979323846*t/2);
  }
  void comp_context(double ratio,bool local,bool real){
    if(++stamp_==0){std::fill(marks_.begin(),marks_.end(),0);stamp_=1;}unique_anchors_.clear();
    for(int64_t id:selected_){int a=owners()[id];if(marks_[a]!=stamp_){marks_[a]=stamp_;unique_anchors_.push_back(a);}}
    int full,partial;float weight,code[3];controls(ratio,full,partial,weight,code);alignas(32) float rp[64];ratio_.apply(code,1,3,rp,H_);
    pool_.run(unique_anchors_.size(),64,[&](int begin,int end,int){
      alignas(32) float condition[8],vp[64];
      for(int i=begin;i<end;++i){int a=unique_anchors_[i];std::fill_n(vp,H_,0.f);
        if(schema_&&local){for(int j=0;j<7;++j)condition[j]=real?view_code_[a*7+j]:0.f;condition[7]=1.f;view_.apply(condition,1,8,vp,H_);}
        float* out=comp_dynamic_.data()+a*H_;
        for(int o=0;o<H_;o+=8){__m256 v=_mm256_loadu_ps(pprefix_+(int64_t(full)*A_+a)*H_+o);
          if(partial>=0)v=_mm256_fmadd_ps(_mm256_set1_ps(weight),_mm256_loadu_ps(pfeature_+(int64_t(partial)*A_+a)*H_+o),v);
          v=_mm256_add_ps(v,_mm256_load_ps(rp+o));v=_mm256_add_ps(v,_mm256_load_ps(vp+o));_mm256_storeu_ps(out+o,v);}
      }
    });
  }
  void publish(int row,int id,const float* delta,float* output,bool compensate){
    int64_t k=K_;float* xyz=output;float* op=xyz+k*3;float* scale=op+k;float* rotation=scale+k*3;float* dc=rotation+k*4;float* rest=dc+k*3;float* means=rest+k*rest_;
    float q[4];for(int j=0;j<4;++j)q[j]=f(6)[id*4+j];
    if(compensate){
      float d[4]={1+delta[6],delta[7],delta[8],delta[9]},norm=0;for(float v:d)norm+=v*v;norm=std::max(std::sqrt(norm),1e-12f);for(float& v:d)v/=norm;
      float v[4]={d[0]*q[0]-((d[1]*q[1]+d[2]*q[2])+d[3]*q[3]),d[0]*q[1]+q[0]*d[1]+(d[2]*q[3]-d[3]*q[2]),d[0]*q[2]+q[0]*d[2]+(d[3]*q[1]-d[1]*q[3]),d[0]*q[3]+q[0]*d[3]+(d[1]*q[2]-d[2]*q[1])};
      for(int pass=0;pass<2;++pass){norm=0;for(float x:v)norm+=x*x;norm=std::max(std::sqrt(norm),1e-12f);for(float& x:v)x/=norm;}std::copy_n(v,4,q);
    }
    for(int j=0;j<3;++j){
      xyz[row*3+j]=f(4)[id*3+j]+(compensate?delta[j]:0.f);float base=activated_scale_[id*3+j],value=base;
      if(compensate){if(softplus_){float v=delta[3+j]+.5413248546129181f;value=base*(v>20?v:std::log1p(std::exp(v)));value=std::max(value,std::numeric_limits<float>::min());}else value=base+delta[3+j];}
      scale[row*3+j]=value;dc[row*3+j]=f(8)[id*3+j]+(compensate?delta[11+j]:0.f);means[row*3+j]=0;
      if(!(value>0)||!std::isfinite(value)||!std::isfinite(xyz[row*3+j])||!std::isfinite(dc[row*3+j]))bad_.store(3,std::memory_order_relaxed);
    }
    float logit=f(7)[id]+(compensate?delta[10]:0.f);op[row]=compensate?1.f/(1.f+std::exp(-logit)):activated_opacity_[id];
    std::copy_n(q,4,rotation+row*4);std::copy_n(f(9)+id*rest_,rest_,rest+row*rest_);
    for(float v:q)if(!std::isfinite(v))bad_.store(3,std::memory_order_relaxed);if(!std::isfinite(op[row]))bad_.store(3,std::memory_order_relaxed);
  }
  void compensate(double ratio,bool local,bool real,float* output,bool enabled){
    if(!enabled){pool_.run(K_,128,[&](int begin,int end,int){for(int i=begin;i<end;++i)publish(i,selected_[i],nullptr,output,false);});return;}
    comp_context(ratio,local,real);
    int full,partial;float weight,code[3];controls(ratio,full,partial,weight,code);
    pool_.run((K_+3)/4,16,[&](int begin,int end,int){
      alignas(32) float h0[4][64],h1[4][64],hidden[4][256],small[4][4],delta[4][14],input[4][56],rp[64],vp[64];
      for(int block=begin;block<end;++block){int base=block*4,R=std::min(4,K_-base);
        if(factored_){for(int r=0;r<R;++r){int id=selected_[base+r],a=owners()[id];for(int o=0;o<H_;o+=8){__m256 v=_mm256_add_ps(_mm256_loadu_ps(poffset_+id*H_+o),_mm256_loadu_ps(comp_dynamic_.data()+a*H_+o));_mm256_store_ps(h0[r]+o,_mm256_max_ps(_mm256_setzero_ps(),v));}}}
        else{
          for(int r=0;r<R;++r){int id=selected_[base+r],a=owners()[id];std::copy_n(f(11)+id*3,3,input[r]);
            for(int j=0;j<48;++j){float v=table_ptrs_[0][a*48+j];for(int l=1;l<=full;++l)v+=table_ptrs_[l][a*48+j];if(partial>=0)v+=weight*table_ptrs_[partial][a*48+j];input[r][j+3]=v;}}
          comp0_.apply(input[0],R,56,h0[0],64);ratio_.apply(code,1,3,rp,H_);
          for(int r=0;r<R;++r){std::fill_n(vp,H_,0.f);if(schema_&&local){float cond[8];int a=owners()[selected_[base+r]];for(int j=0;j<7;++j)cond[j]=real?view_code_[a*7+j]:0;cond[7]=1;view_.apply(cond,1,8,vp,H_);}
            for(int o=0;o<H_;o+=8){__m256 v=_mm256_add_ps(_mm256_load_ps(h0[r]+o),_mm256_load_ps(rp+o));v=_mm256_add_ps(v,_mm256_load_ps(vp+o));_mm256_store_ps(h0[r]+o,_mm256_max_ps(_mm256_setzero_ps(),v));}}
        }
        comp1_.apply(h0[0],R,64,h1[0],64,true);
        if(!student_)combined_head_.apply(h1[0],R,64,hidden[0],256,true);
        for(int head=0;head<4;++head){const float* source=student_?h1[0]:hidden[0]+head*64;
          ho_[head].apply(source,R,student_?64:256,small[0],4);int offset=head==0?0:head==1?3:head==2?6:10;int count=head<2?3:4;
          for(int r=0;r<R;++r)for(int j=0;j<count;++j)delta[r][offset+j]=small[r][j];}
        for(int r=0;r<R;++r){int id=selected_[base+r];float reference=f(1)[owners()[id]];for(int j=0;j<3;++j)delta[r][j]*=reference;
          bool finite=true;for(float v:delta[r])finite=finite&&std::isfinite(v);if(!finite)bad_.store(3,std::memory_order_relaxed);
          if(diagnose_)std::copy_n(delta[r],14,residual_.data()+(base+r)*14);publish(base+r,id,delta[r],output,true);}
      }
    });
  }
public:
  CpuPipelineDeep(std::vector<T> state,std::vector<T> selector,std::vector<T> comp,std::vector<T> tables,T view_weight,
      std::vector<double> knots,std::vector<double> bounds,int schema,float scene_scale,bool softplus,int threads,
      bool selector_prefix=true,bool factored=true,bool diagnose=false,bool fused_schedule=false,int kernel_mode=0)
      :fused_schedule_(fused_schedule),kernel_mode_(kernel_mode),pool_(threads),knots_(knots),bounds_(bounds),student_(comp.size()==13),softplus_(softplus),selector_prefix_(selector_prefix),factored_(factored),diagnose_(diagnose),schema_(schema),scene_scale_(scene_scale){
    TORCH_CHECK(state.size()==13&&selector.size()==6&&(comp.size()==21||student_),"Unsupported CPU snapshot layout");
    TORCH_CHECK(schema>=0&&schema<=4&&std::isfinite(scene_scale)&&scene_scale>0,"Joint view schema/scale");
    for(int i=0;i<13;++i)s_.push_back(own(state[i],i==2?at::kLong:i==3?at::kBool:at::kFloat));
    for(int i=0;i<13;++i)if(i!=2&&i!=3)F_[i]=s_[i].data_ptr<float>();owner_ptr_=s_[2].data_ptr<int64_t>();eligible_ptr_=s_[3].data_ptr<bool>();
    N_=s_[4].size(0);A_=s_[0].size(0);H_=comp[0].size(0);rest_=s_[9].numel()/std::max(1,N_);width_=17+rest_;
    TORCH_CHECK(N_>0&&N_<std::numeric_limits<int>::max()&&A_>0&&(H_==32||H_==64)&&bounds.size()==2&&knots.size()>=2&&knots.size()<=6,"CPU snapshot dimensions");
    TORCH_CHECK(s_[0].numel()==A_*3&&s_[1].numel()==A_&&s_[2].numel()==N_&&s_[3].numel()==N_&&s_[4].numel()==N_*3&&s_[5].numel()==N_*3&&s_[6].numel()==N_*4&&s_[7].numel()==N_&&s_[8].numel()==N_*3&&s_[10].numel()==N_*10&&s_[11].numel()==N_*3&&s_[12].numel()==N_*2,"CPU state shapes");
    TORCH_CHECK(tables.size()==knots.size(),"Residual table count");for(auto& t:tables){TORCH_CHECK(t.numel()==A_*48,"Residual48 shape");tables_.push_back(own(t));}
    sel0_=Dense(selector[0],selector[1]);sel1_=Dense(selector[2],selector[3]);sel2_=Dense(selector[4],selector[5]);
    TORCH_CHECK(sel0_.in==(schema?19:11)&&sel0_.out==32&&sel1_.in==32&&sel1_.out==32&&sel2_.out==2,"Selector shape");
    comp0_=Dense(comp[0],comp[1]);comp1_=Dense(comp[2],comp[3]);ratio_=Dense(comp[4],at::zeros({H_},comp[4].options()));
    TORCH_CHECK(comp0_.in==51&&comp1_.in==H_&&comp1_.out==H_&&ratio_.in==3,"Compensator shape");
    int p=5;for(int head=0;head<4;++head){if(!student_){hh_.emplace_back(comp[p],comp[p+1]);p+=2;}ho_.emplace_back(comp[p],comp[p+1]);p+=2;TORCH_CHECK(ho_.back().out==(head<2?3:4),"Head outputs");}
    if(!student_){std::vector<T> ws,bs;for(int head=0;head<4;++head){ws.push_back(comp[5+head*4]);bs.push_back(comp[6+head*4]);}combined_head_=Dense(at::cat(ws,0),at::cat(bs,0));}
    TORCH_CHECK(kernel_mode>=0&&kernel_mode<=1,"Unknown AVX2 kernel mode");
    sel0_.fast=sel1_.fast=sel2_.fast=comp0_.fast=comp1_.fast=ratio_.fast=combined_head_.fast=kernel_mode==1;
    for(auto& h:ho_)h.fast=kernel_mode==1;
    activated_scale_.resize(N_*3);activated_opacity_.resize(N_);
    pool_.run(N_,128,[&](int begin,int end,int){for(int id=begin;id<end;++id){for(int j=0;j<3;++j)activated_scale_[id*3+j]=std::exp(f(5)[id*3+j]);activated_opacity_[id]=1.f/(1.f+std::exp(-f(7)[id]));}});
    if(schema_){view_=Dense(view_weight,at::zeros({H_},view_weight.options()));view_.fast=kernel_mode==1;}
    auto options=s_[4].options();int64_t estimate=int64_t(N_)*(32+H_)*4+int64_t(A_)*H_*tables.size()*8;
    TORCH_CHECK(estimate<1024ll*1024*1024,"Frozen CPU projected cache exceeds 1 GiB");
    selector_prefix_data_=at::empty({N_,32},options);offset_prefix_=at::empty({N_,H_},options);
    feature_projected_=at::empty({int64_t(tables.size()),A_,H_},options);feature_prefix_=at::empty_like(feature_projected_);
    pselector_=selector_prefix_data_.data_ptr<float>();poffset_=offset_prefix_.data_ptr<float>();pfeature_=feature_projected_.data_ptr<float>();pprefix_=feature_prefix_.data_ptr<float>();
    for(auto& t:tables_)table_ptrs_.push_back(t.data_ptr<float>());
    occupied_.assign(A_,0);visible_.resize(A_);eligible_counts_.assign(A_,0);centers_soa_.resize(A_*4);corners_.resize(A_*24);
    marks_.assign(A_,0);view_code_.resize(A_*7);sel_dynamic_.resize(A_*32);comp_dynamic_.resize(A_*H_);
    rows_.resize(N_);scores_.resize(N_);keys_.resize(N_);selected_.reserve(N_);unique_anchors_.reserve(A_);
    if(diagnose_)residual_.resize(N_*14);counts_.resize(threads);offsets_.resize(threads);
    for(int i=0;i<N_;++i){TORCH_CHECK(owners()[i]>=0&&owners()[i]<A_,"Anchor owner outside table");occupied_[owners()[i]]=1;eligible_counts_[owners()[i]]+=eligible()[i];}
    for(int a=0;a<A_;++a){TORCH_CHECK(f(1)[a]>0,"Nonpositive reference scale");for(int j=0;j<3;++j)centers_soa_[j*A_+a]=f(0)[a*3+j];centers_soa_[3*A_+a]=f(1)[a]*.5f;
      for(int c=0;c<8;++c)for(int j=0;j<3;++j)corners_[a*24+j*8+c]=f(0)[a*3+j]+f(1)[a]*.5f*((c&(1<<(2-j)))?1.f:-1.f);}
    pool_.run(N_,128,[&](int begin,int end,int){for(int id=begin;id<end;++id){
      for(int o=0;o<32;o+=8){__m256 v=_mm256_setzero_ps();for(int j=0;j<10;++j)v=_mm256_fmadd_ps(_mm256_set1_ps(f(10)[id*10+j]),_mm256_loadu_ps(sel0_.weights_ptr+(o/8)*sel0_.in*8+j*8),v);_mm256_storeu_ps(pselector_+id*32+o,v);}
      for(int o=0;o<H_;o+=8){__m256 v=_mm256_setzero_ps();for(int j=0;j<3;++j)v=_mm256_fmadd_ps(_mm256_set1_ps(f(11)[id*3+j]),_mm256_loadu_ps(comp0_.weights_ptr+(o/8)*51*8+j*8),v);v=_mm256_add_ps(v,_mm256_loadu_ps(comp0_.bias_ptr+o));_mm256_storeu_ps(poffset_+id*H_+o,v);}
    }});
    pool_.run(A_,64,[&](int begin,int end,int){for(int a=begin;a<end;++a)for(int level=0;level<int(tables.size());++level)for(int o=0;o<H_;o+=8){
      __m256 v=_mm256_setzero_ps();for(int j=0;j<48;++j)v=_mm256_fmadd_ps(_mm256_set1_ps(tables_[level].data_ptr<float>()[a*48+j]),_mm256_loadu_ps(comp0_.weights_ptr+(o/8)*51*8+(j+3)*8),v);
      int64_t offset=(int64_t(level)*A_+a)*H_+o;_mm256_storeu_ps(pfeature_+offset,v);
      if(level)v=_mm256_add_ps(_mm256_loadu_ps(pprefix_+offset-int64_t(A_)*H_),v);_mm256_storeu_ps(pprefix_+offset,v);
    }});
    for(auto& t:s_)bytes_+=t.nbytes();for(auto& t:tables_)bytes_+=t.nbytes();for(auto& t:{selector_prefix_data_,offset_prefix_,feature_projected_,feature_prefix_})bytes_+=t.nbytes();
    bytes_+=int64_t(centers_soa_.size()+corners_.size()+view_code_.size()+sel_dynamic_.size()+comp_dynamic_.size()+scores_.size()+residual_.size())*4;
    bytes_+=int64_t(activated_scale_.size()+activated_opacity_.size())*4;
    bytes_+=int64_t(N_)*(sizeof(int64_t)*2+sizeof(Key))+int64_t(A_)*(sizeof(int)*2+sizeof(uint32_t)+2);
  }
  std::tuple<T,T,std::vector<double>,std::vector<double>> prepare(T vm,T pm,T camera,int width,int height,float near,
      int kind,double ratio,int budget,bool local=true,bool compensation=true,bool timing=false,
      std::optional<T> output_arg=std::nullopt,std::optional<T> id_arg=std::nullopt){
    std::lock_guard<std::mutex> lock(call_);auto t0=Clock::now();std::vector<double> times;bad_=0;
    T output=output_arg.value_or(T()),id_output=id_arg.value_or(T());
    TORCH_CHECK(vm.device().is_cpu()&&pm.device().is_cpu()&&camera.device().is_cpu()&&vm.scalar_type()==at::kFloat&&pm.scalar_type()==at::kFloat&&camera.scalar_type()==at::kFloat&&vm.is_contiguous()&&pm.is_contiguous()&&camera.is_contiguous()&&vm.numel()==16&&pm.numel()==16&&camera.numel()==3,"CPU camera inputs");
    TORCH_CHECK(kind>=0&&kind<=3&&width>0&&height>0&&std::isfinite(near)&&near>0,"CPU request dimensions");
    for(auto& t:{vm,pm,camera})for(int i=0;i<t.numel();++i)TORCH_CHECK(std::isfinite(t.data_ptr<float>()[i]),"Nonfinite camera");
    if(kind==0||kind==2)TORCH_CHECK(std::isfinite(ratio)&&ratio>=0&&ratio<=1&&(kind!=2||ratio>0),"Invalid ratio");
    if(kind==1)TORCH_CHECK(budget>=0,"Invalid budget");
    bool global=kind>=2;if(global)local=false;
    auto checkpoint=[&]{if(timing){auto now=Clock::now();times.push_back(ms(t0,now));t0=now;}};
    bool context=fused_schedule_&&kind!=3&&((kind==1&&budget>0)||(kind!=1&&ratio>0));
    frustum(pm.data_ptr<float>(),global,vm.data_ptr<float>(),camera.data_ptr<float>(),float(width)/height,near,local,compensation,context);candidates(global);int visible=0,available=0;for(int a=0;a<A_;++a)if(visible_[a]){++visible;available+=eligible_counts_[a];}
    if(kind==1)ratio=M_&&budget?std::min(1.,double(budget)/M_):0.;
    int requested=kind==1?budget:kind==3?N_:kind==2?(M_?M_-int((1-ratio)*(M_-1)):0):int(std::floor(ratio*M_));
    K_=std::min(requested,available);selected_.clear();checkpoint();
    if(kind==3){for(int i=0;i<N_;++i)selected_.push_back(i);K_=N_;compensation=false;}
    else if(K_){
      bool selector_real=schema_==2||schema_==3,comp_real=schema_==2||schema_==4;
      if(!fused_schedule_)pool_.run(A_,128,[&](int begin,int end,int){for(int a=begin;a<end;++a)if(visible_[a])anchor_context(a,vm.data_ptr<float>(),pm.data_ptr<float>(),camera.data_ptr<float>(),float(width)/height,near,local,compensation);});checkpoint();
      keys_.resize(M_);selector(float(std::log(ratio)),local,selector_real);TORCH_CHECK(!bad_.load(),"Nonfinite view/Selector output");checkpoint();
      if(!fused_schedule_)pool_.run(M_,1024,[&](int begin,int end,int){for(int i=begin;i<end;++i)keys_[i]={eligible()[rows_[i]]?scores_[i]:-INFINITY,uint32_t(rows_[i])};});
      if(kind==2){int rank=int((1-ratio)*(M_-1));float threshold=-INFINITY;
        if(rank>=0){std::nth_element(keys_.begin(),keys_.begin()+rank,keys_.end(),[](const Key& a,const Key& b){return a.score<b.score;});threshold=keys_[rank].score;}
        for(auto& k:keys_)if(eligible()[k.id]&&k.score>=threshold)selected_.push_back(k.id);K_=selected_.size();
      }else{if(K_<int(keys_.size()))std::nth_element(keys_.begin(),keys_.begin()+K_,keys_.end(),Better{});for(int i=0;i<K_;++i)selected_.push_back(keys_[i].id);}
      std::sort(selected_.begin(),selected_.end());checkpoint();
    }else{checkpoint();checkpoint();checkpoint();}
    int64_t size=int64_t(K_)*width_;auto options=s_[4].options();T flat,ids;
    if(output.defined()){TORCH_CHECK(output.device().is_cpu()&&output.scalar_type()==at::kFloat&&output.is_contiguous()&&output.numel()>=size,"CPU output capacity/type");flat=output.flatten().narrow(0,0,size);}else flat=at::empty({size},options);
    if(id_output.defined()){TORCH_CHECK(id_output.device().is_cpu()&&id_output.scalar_type()==at::kLong&&id_output.is_contiguous()&&id_output.numel()>=K_,"CPU ID output capacity/type");ids=id_output.flatten().narrow(0,0,K_);}else ids=at::empty({K_},options.dtype(at::kLong));
    if(K_){std::copy(selected_.begin(),selected_.end(),ids.data_ptr<int64_t>());compensate(ratio,local,schema_==2||schema_==4,flat.data_ptr<float>(),compensation);}
    TORCH_CHECK(!bad_.load(),"Nonfinite or nonpositive compensated Renderer attributes");checkpoint();
    return {flat,ids,{double(N_),double(A_),double(visible),double(M_),double(available),double(requested),double(K_),ratio,double(bytes_),double(pool_.count())},times};
  }
  std::vector<T> diagnostics(){std::lock_guard<std::mutex> lock(call_);auto options=s_[4].options();auto v=at::empty({A_},options.dtype(at::kBool)),r=at::empty({M_},options.dtype(at::kLong)),score=at::empty({M_},options);std::copy(visible_.begin(),visible_.end(),v.data_ptr<bool>());std::copy_n(rows_.data(),M_,r.data_ptr<int64_t>());std::copy_n(scores_.data(),M_,score.data_ptr<float>());
    auto residual=at::empty({diagnose_?K_:0,14},options);if(diagnose_)std::copy_n(residual_.data(),K_*14,residual.data_ptr<float>());return {v,r,score,residual};}
  std::vector<T> packet_views(const T& flat,int64_t k)const{
    TORCH_CHECK(flat.dim()==1&&flat.numel()==k*width_,"Packet size");
    std::vector<T> v;int64_t p=0;
    for(int w:{3,1,3,4,3,rest_,3}){auto t=flat.narrow(0,p,k*w);p+=k*w;
      if(v.size()==4)v.push_back(t.view({k,1,3}));else if(v.size()==5)v.push_back(t.view({k,rest_/3,3}));else v.push_back(t.view({k,w}));}
    return v;
  }
  int64_t total()const{return N_;}int64_t width()const{return width_;}
  int64_t bytes()const{return bytes_;}
};

#include "native_handoff.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  pybind11::class_<CpuPipelineDeep,std::shared_ptr<CpuPipelineDeep>>(m,"CpuPipeline")
    .def(pybind11::init<std::vector<T>,std::vector<T>,std::vector<T>,std::vector<T>,T,std::vector<double>,std::vector<double>,int,float,bool,int,bool,bool,bool,bool,int>(),
      pybind11::arg("state"),pybind11::arg("selector"),pybind11::arg("compensator"),pybind11::arg("tables"),pybind11::arg("view_weight"),pybind11::arg("knots"),pybind11::arg("bounds"),pybind11::arg("schema"),pybind11::arg("scene_scale"),pybind11::arg("softplus"),pybind11::arg("threads"),pybind11::arg("selector_prefix")=true,pybind11::arg("factored")=true,pybind11::arg("diagnose")=false,pybind11::arg("fused_schedule")=false,pybind11::arg("kernel_mode")=0)
    .def("prepare",&CpuPipelineDeep::prepare,pybind11::call_guard<pybind11::gil_scoped_release>(),pybind11::arg("view"),pybind11::arg("projection"),pybind11::arg("camera"),pybind11::arg("width"),pybind11::arg("height"),pybind11::arg("near"),pybind11::arg("kind"),pybind11::arg("ratio"),pybind11::arg("budget"),pybind11::arg("local")=true,pybind11::arg("compensation")=true,pybind11::arg("timing")=false,pybind11::arg("output")=pybind11::none(),pybind11::arg("id_output")=pybind11::none())
    .def("prepare_packet",&prepare_cpu,pybind11::arg("camera"),pybind11::arg("kind"),pybind11::arg("ratio"),pybind11::arg("budget"),pybind11::arg("local"),pybind11::arg("compensation"),pybind11::arg("timing"),pybind11::arg("output"),pybind11::arg("id_output"),pybind11::arg("sh_degree"))
    .def("packet_views",&CpuPipelineDeep::packet_views).def("diagnostics",&CpuPipelineDeep::diagnostics).def("workspace_bytes",&CpuPipelineDeep::bytes);
  bind_handoff(m);
}
