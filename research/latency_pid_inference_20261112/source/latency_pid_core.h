#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace latency_pid {
struct Config {
  double target_ms=.4, kp=1., ki=4., kd=0.;
  double ema=.25, derivative_ema=.2, deadband=.05;
  double max_increase=.08, max_decrease=.25, max_dt=.1;
  double cut_translation=.5, cut_rotation_degrees=30.;
  int64_t initial_budget=3000, min_budget=256, max_budget=8192;

  void validate() const {
    for(double v:{target_ms,kp,ki,kd,ema,derivative_ema,deadband,
        max_increase,max_decrease,max_dt,cut_translation,cut_rotation_degrees})
      if(!std::isfinite(v))throw std::invalid_argument("PID options must be finite");
    if(target_ms<=0 || kp<0 || ki<0 || kd<0 || ema<=0 || ema>1 ||
       derivative_ema<=0 || derivative_ema>1 || deadband<0 || deadband>=1 ||
       max_increase<=0 || max_decrease<=0 || max_decrease>=1 || max_dt<=0 ||
       cut_translation<=0 || cut_rotation_degrees<=0 || cut_rotation_degrees>180 ||
       min_budget<1 || max_budget<min_budget || initial_budget<min_budget || initial_budget>max_budget)
      throw std::invalid_argument("Invalid PID range or budget bounds");
  }
};

// CPU scalar controller only. GPU candidate controls retain the final strict-K clamp.
class Core {
  double base_log_, desired_log_, last_time_=-1., previous_measurement_=0.;
  bool filtered_=false;
public:
  Config config;
  double filtered_ms=0., integral=0., derivative=0.;
  int64_t applied_k=0, updates=0;

  explicit Core(Config value):config(value) {
    config.validate();base_log_=desired_log_=std::log(double(config.initial_budget));
  }
  int64_t budget() const {
    return std::max(config.min_budget,std::min(config.max_budget,int64_t(std::llround(std::exp(desired_log_)))));
  }
  void applied(int64_t k) {
    if(k<0 || k>config.max_budget)throw std::invalid_argument("Invalid effective PID budget");
    applied_k=k;
  }
  void reset() {
    base_log_=desired_log_=std::log(double(std::max(config.min_budget,
        std::min(config.max_budget,applied_k?applied_k:budget()))));
    filtered_=false;last_time_=-1.;filtered_ms=integral=derivative=previous_measurement_=0.;
  }
  void target(double ms) {
    if(!std::isfinite(ms) || ms<=0)throw std::invalid_argument("PID target must be positive and finite");
    reset();config.target_ms=ms;
  }
  void observe(double ms,double submitted_s,int64_t sample_k,int64_t available) {
    if(!std::isfinite(ms) || ms<=0 || !std::isfinite(submitted_s) ||
        sample_k<0 || available<sample_k || sample_k>config.max_budget)
      throw std::invalid_argument("Invalid PID timing sample");
    if(last_time_>=0 && submitted_s<=last_time_)throw std::invalid_argument("PID samples must be chronological");
    const double dt=last_time_<0?0.:std::min(config.max_dt,submitted_s-last_time_);
    last_time_=submitted_s;
    if(!sample_k || !available){filtered_=false;integral=derivative=0.;return;}
    const double normalized=ms/config.target_ms;
    if(!filtered_){filtered_ms=ms;previous_measurement_=normalized;derivative=0.;filtered_=true;}
    else {
      filtered_ms+=config.ema*(ms-filtered_ms);
      const double measurement=filtered_ms/config.target_ms;
      const double slope=dt>0?(measurement-previous_measurement_)/dt:0.;
      derivative+=config.derivative_ema*(slope-derivative);
      previous_measurement_=measurement;
    }
    const double error=(config.target_ms-filtered_ms)/config.target_ms;
    ++updates;
    // Within the acceptable interval, hold the last actual budget and do not wind up I.
    if(std::abs(error)<=config.deadband){
      desired_log_=std::log(double(std::max(config.min_budget,
          std::min(config.max_budget,applied_k?applied_k:sample_k))));return;
    }
    const int64_t high=std::min(config.max_budget,available),low=std::min(config.min_budget,high);
    const double lo=std::log(double(std::max(int64_t(1),low))),hi=std::log(double(std::max(int64_t(1),high)));
    const double trial=integral+error*dt;
    double command=base_log_+config.kp*error+config.ki*trial-config.kd*derivative;
    // Conditional integration: a shortage or bound cannot keep charging the integrator.
    if(!((command>hi && error>0) || (command<lo && error<0)))integral=trial;
    command=base_log_+config.kp*error+config.ki*integral-config.kd*derivative;
    command=std::max(lo,std::min(hi,command));
    const double current=std::log(double(std::max(int64_t(1),applied_k?applied_k:sample_k)));
    desired_log_=std::max(current+std::log1p(-config.max_decrease),
        std::min(current+std::log1p(config.max_increase),command));
    desired_log_=std::max(lo,std::min(hi,desired_log_));
  }
};
}
