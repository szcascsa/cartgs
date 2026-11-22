#pragma once
#include "gaussian_frame.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace mycartgs {
struct CpuBenchmarkCase { float ratio;uint32_t fixedK; };
struct CpuBenchmarkSample {
    uint32_t stage,iteration;bool warmup;CpuBenchmarkCase budget;FrameStatistics stats;
    double totalMs;uint32_t selectorCalls,studentCalls;
};
class CpuBenchmark {
public:
    static constexpr uint32_t kWarmup=5,kMeasured=30;
    void Start(){plan_={{.1f,0},{.2f,0},{.3f,0},{.4f,0},{.5f,0},{.1f,1000},{.1f,2000},{.1f,5000},{.1f,10000}};
        stage_=iteration_=0;samples_.clear();samples_.reserve(plan_.size()*(kWarmup+kMeasured));running_=true;}
    bool Running()const{return running_;}
    void Cancel(){running_=false;}
    uint32_t Stage()const{return stage_;}
    uint32_t Iteration()const{return iteration_;}
    uint32_t Cases()const{return uint32_t(plan_.size());}
    const CpuBenchmarkCase& Budget()const{return plan_.at(stage_);}
    const std::vector<CpuBenchmarkSample>& Samples()const{return samples_;}
    void Record(const FrameStatistics& s,double total,uint32_t selector,uint32_t student){
        if(!running_||!std::isfinite(total)||total<0||selector>1||student>1)throw std::runtime_error("Invalid CPU benchmark sample");
        for(double t:{s.prepareMs,s.selectorMs,s.topkMs,s.studentMs,s.decodeMs})if(!std::isfinite(t)||t<0)throw std::runtime_error("Invalid CPU stage time");
        samples_.push_back({stage_,iteration_,iteration_<kWarmup,Budget(),s,total,selector,student});
        if(++iteration_==kWarmup+kMeasured){iteration_=0;if(++stage_==plan_.size())running_=false;}
    }
    static double Percentile(std::vector<double> values,double p){
        if(values.empty())return 0;std::sort(values.begin(),values.end());const auto n=values.size();
        return p==.5?(n%2?values[n/2]:(values[n/2-1]+values[n/2])*.5):values[size_t(std::ceil(p*n))-1];}
    std::vector<std::string> Rows()const {
        std::vector<std::string> result;
        for(uint32_t stage=0;stage<plan_.size();++stage){std::vector<double> total,network;uint32_t n=0,k=0;
            for(const auto& s:samples_)if(s.stage==stage&&!s.warmup){total.push_back(s.totalMs);network.push_back(s.stats.selectorMs+s.stats.studentMs);n=s.stats.candidates;k=s.stats.k;}
            if(total.empty())continue;char line[112],label[20];if(plan_[stage].fixedK)std::snprintf(label,sizeof(label),"K%u",plan_[stage].fixedK);
            else std::snprintf(label,sizeof(label),"R%u",unsigned(std::lround(plan_[stage].ratio*100)));
            std::snprintf(line,sizeof(line),"%-7s %-6u %-6u %6.2f %6.2f %6.2f",label,n,k,Percentile(total,.5),Percentile(total,.95),Percentile(network,.5));result.emplace_back(line);}
        return result;
    }
    std::string Csv()const {
        std::ostringstream out;out.precision(10);out<<"case,ratio,fixed_k,conditioning_ratio,iteration,warmup,candidates,eligible,k,prepare_ms,selector_ms,topk_ms,student_ms,decode_ms,network_ms,cpu_total_ms,selector_calls,student_calls\n";
        for(const auto& s:samples_){const auto& t=s.stats;const double condition=s.budget.fixedK?(t.candidates?std::min(1.,double(s.budget.fixedK)/t.candidates):0.):s.budget.ratio;
            out<<s.stage<<','<<s.budget.ratio<<','<<s.budget.fixedK<<','<<condition<<','<<s.iteration<<','<<s.warmup<<','<<t.candidates<<','<<t.eligible<<','<<t.k<<','
            <<t.prepareMs<<','<<t.selectorMs<<','<<t.topkMs<<','<<t.studentMs<<','<<t.decodeMs<<','<<t.selectorMs+t.studentMs<<','<<s.totalMs<<','<<s.selectorCalls<<','<<s.studentCalls<<'\n';}
        return out.str();
    }
private:
    bool running_=false;uint32_t stage_=0,iteration_=0;std::vector<CpuBenchmarkCase> plan_;std::vector<CpuBenchmarkSample> samples_;
};
}
