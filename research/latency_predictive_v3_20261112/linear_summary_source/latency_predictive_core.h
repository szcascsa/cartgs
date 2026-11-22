#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace latency_predictive {
struct View { int64_t available=0; double footprint=1.; };

// 四维非负加权最小二乘；只学习过去实际请求的GPU耗时。
class Core {
  std::array<double,4> theta_;
  std::array<std::array<double,4>,4> gram_{};
  std::array<double,4> cross_{};
  double last_time_=-1.,bias_=0.,last_ms_=0.;
  int64_t last_k_=0;
  View last_view_;
  void fit(){
    // 4变量枚举16个活跃集，避免相关输入下投影RLS破坏协方差语义。
    double best=1e100;std::array<double,4> chosen{};
    for(int mask=0;mask<16;++mask){
      int index[4],n=0;for(int i=0;i<4;++i)if(mask&(1<<i))index[n++]=i;
      double a[4][5]{};for(int i=0;i<n;++i){for(int j=0;j<n;++j)a[i][j]=gram_[index[i]][index[j]];
        a[i][i]+=.0001;a[i][n]=cross_[index[i]];}
      bool valid=true;
      for(int c=0;c<n;++c){int pivot=c;for(int r=c+1;r<n;++r)if(std::abs(a[r][c])>std::abs(a[pivot][c]))pivot=r;
        if(std::abs(a[pivot][c])<1e-12){valid=false;break;}
        if(pivot!=c)for(int j=0;j<=n;++j)std::swap(a[pivot][j],a[c][j]);
        const double d=a[c][c];for(int j=c;j<=n;++j)a[c][j]/=d;
        for(int r=0;r<n;++r)if(r!=c){const double v=a[r][c];for(int j=c;j<=n;++j)a[r][j]-=v*a[c][j];}}
      if(!valid)continue;
      std::array<double,4> t{};for(int i=0;i<n;++i){t[index[i]]=a[i][n];valid=valid&&a[i][n]>=0.;}
      if(!valid)continue;double error=0.;
      for(int i=0;i<4;++i){error-=2*t[i]*cross_[i];for(int j=0;j<4;++j)error+=t[i]*gram_[i][j]*t[j];error+=.0001*t[i]*t[i];}
      if(error<best){best=error;chosen=t;}
    }
    theta_=chosen;
  }
public:
  double target_ms=.2,max_increase=.35,max_decrease=.4;
  int64_t initial_budget=3000,min_budget=256,max_budget=43197,updates=0,model_resets=0;
  double filtered_ms=0.,integral=0.,derivative=0.,predicted_ms=0.;
  int64_t applied_k=0,predicted_available=0;
  double current_footprint=1.;bool pool_limited=false;

  explicit Core(double target,int64_t initial,int64_t low,int64_t high):
      target_ms(target),initial_budget(initial),min_budget(low),max_budget(high) {
    if(!std::isfinite(target)||target<=0||low<1||high<low||initial<low||initial>high)
      throw std::invalid_argument("Invalid predictive control bounds");
    reset_model();
  }
  void reset_model(){
    theta_={target_ms*.35,target_ms*.2,target_ms*.1,target_ms*.1};
    for(int i=0;i<4;++i){cross_[i]=.01*theta_[i];for(int j=0;j<4;++j)gram_[i][j]=i==j?.01:0.;}
    last_time_=-1.;bias_=last_ms_=0.;last_k_=0;filtered_ms=0.;
  }
  void reset(){last_time_=-1.;last_ms_=0.;bias_=0.;last_k_=0;filtered_ms=0.;}
  void target(double ms){
    if(!std::isfinite(ms)||ms<=0)throw std::invalid_argument("Invalid predictive target");
    target_ms=ms;reset_model();
  }
  void applied(int64_t k){
    if(k<0||k>max_budget)throw std::invalid_argument("Invalid predictive actual K");
    applied_k=k;
  }
  static std::array<double,4> features(int64_t k,View v){
    const double work=std::sqrt(double(k)/4096.);
    return {1.,work,work*v.footprint,double(k)/std::max(int64_t(1),v.available)};
  }
  double predict(int64_t k,View v)const{
    auto x=features(k,v);double y=0.;for(int i=0;i<4;++i)y+=theta_[i]*x[i];return y+bias_;
  }
  void observe(double ms,double time,int64_t k,int64_t available,View view){
    if(!std::isfinite(ms)||ms<=0||!std::isfinite(time)||(last_time_>=0&&time<=last_time_)||
        k<0||available<k||k>max_budget||!std::isfinite(view.footprint)||view.footprint<0.)
      throw std::invalid_argument("Invalid predictive sample");
    if(last_k_&&std::abs(double(k-last_k_))<=std::max(1.,last_k_*.01)&&
        std::abs(view.footprint-last_view_.footprint)<.03&&std::abs(ms-last_ms_)>target_ms*.3){
      reset_model();++model_resets;
    }
    last_time_=time;
    if(!k||!available){last_k_=0;return;}
    view.available=available;const auto x=features(k,view);
    for(int i=0;i<4;++i){cross_[i]=.97*cross_[i]+x[i]*ms;
      for(int j=0;j<4;++j)gram_[i][j]=.97*gram_[i][j]+x[i]*x[j];}
    fit();
    double fitted=0.;for(int i=0;i<4;++i)fitted+=theta_[i]*x[i];
    bias_=.8*bias_+.2*(ms-fitted);
    bias_=std::max(-target_ms*.25,std::min(target_ms*.25,bias_));
    filtered_ms=updates?filtered_ms+.5*(ms-filtered_ms):ms;
    last_ms_=ms;last_k_=k;last_view_=view;++updates;
  }
  int64_t budget(View view){
    if(view.available<0||!std::isfinite(view.footprint)||view.footprint<0.)
      throw std::invalid_argument("Invalid predictive view");
    predicted_available=view.available;current_footprint=view.footprint;
    const int64_t high=std::max(min_budget,std::min(max_budget,view.available));
    int64_t wanted=std::min(initial_budget,high);
    pool_limited=false;
    if(last_k_){
      pool_limited=predict(high,view)<target_ms;
      int64_t lo=min_budget,hi=high;
      while(lo<hi){const int64_t middle=(lo+hi+1)/2;
        if(predict(middle,view)<=target_ms)lo=middle;else hi=middle-1;}
      wanted=lo;
      const int64_t current=std::max(min_budget,std::min(high,last_k_));
      wanted=std::max(int64_t(std::ceil(current*(1-max_decrease))),
          std::min(int64_t(std::floor(current*(1+max_increase))),wanted));
      // 单次明显超时立即退避；仅在视角几何摘要近似不变时使用旧帧的快速约束。
      if(last_ms_>target_ms*1.05&&std::abs(view.footprint-last_view_.footprint)<.03)
        wanted=std::min(wanted,int64_t(last_k_*std::pow(target_ms/last_ms_,2.)));
    }
    wanted=std::max(min_budget,std::min(high,wanted));predicted_ms=predict(wanted,view);
    return wanted;
  }
};
}
