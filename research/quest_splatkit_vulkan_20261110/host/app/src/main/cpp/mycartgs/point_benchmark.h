#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mycartgs {
struct PointBenchmarkMetrics {
    uint32_t projected=0,activeTiles=0,width=0,height=0;
    uint64_t instances=0;
    double prepareMs=0,uploadMs=0,gpuMs=0,submitWaitMs=0;
    double cullMs=0,sortMs=0,rasterMs=0;
};
struct PointBenchmarkSample {
    uint32_t round=0,k=0,iteration=0;
    bool warmup=false;
    PointBenchmarkMetrics metrics;
};
struct PointBenchmarkSummary {
    uint32_t k=0,projected=0,samples=0;
    double medianMs=0,p95Ms=0;
};

// A bounded, fixed-fixture experiment. No inference result is reused by the normal viewer.
class PointBenchmark {
public:
    static constexpr uint32_t kRounds=2,kWarmup=3,kMeasured=15;
    void Start(uint32_t count) {
        budgets_.clear();samples_.clear();stage_=iteration_=0;
        for(uint32_t k:{0u,1000u,2000u,3000u,4000u,5000u,10000u,15000u,count})
            budgets_.push_back(std::min(k,count));
        std::sort(budgets_.begin(),budgets_.end());
        budgets_.erase(std::unique(budgets_.begin(),budgets_.end()),budgets_.end());
        samples_.reserve(budgets_.size()*kRounds*(kWarmup+kMeasured));running_=true;
    }
    bool Running() const { return running_; }
    void Cancel() { running_=false; }
    uint32_t Stage() const { return stage_; }
    uint32_t Round() const { return budgets_.empty()?0:stage_/uint32_t(budgets_.size()); }
    uint32_t K() const {
        if(!running_)throw std::runtime_error("Point benchmark has no active case");
        const uint32_t index=stage_%uint32_t(budgets_.size());
        return budgets_[Round()%2?budgets_.size()-1-index:index];
    }
    uint32_t Iteration() const { return iteration_; }
    uint32_t TotalCases() const { return uint32_t(budgets_.size())*kRounds; }
    const std::vector<PointBenchmarkSample>& Samples() const { return samples_; }
    void Record(const PointBenchmarkMetrics& m) {
        if(!running_ || !std::isfinite(m.gpuMs) || m.gpuMs<0)
            throw std::runtime_error("Invalid point benchmark sample");
        samples_.push_back({Round(),K(),iteration_,iteration_<kWarmup,m});
        if(++iteration_==kWarmup+kMeasured) {iteration_=0;if(++stage_==TotalCases())running_=false;}
    }
    std::vector<PointBenchmarkSummary> Summaries() const {
        std::vector<PointBenchmarkSummary> result;
        for(uint32_t k:budgets_) {
            std::vector<double> times;uint32_t projected=0;
            for(const auto& s:samples_)if(s.k==k&&!s.warmup){times.push_back(s.metrics.gpuMs);projected=s.metrics.projected;}
            if(times.empty())continue;
            std::sort(times.begin(),times.end());const size_t n=times.size();
            result.push_back({k,projected,uint32_t(n),n%2?times[n/2]:(times[n/2-1]+times[n/2])*.5,
                              times[size_t(std::ceil(.95*n))-1]});
        }
        return result;
    }
    std::string Csv() const {
        std::ostringstream out;out.precision(10);
        out<<"round,k,iteration,warmup,drawn,width,height,prepare_once_ms,upload_once_ms,cull_gpu_ms,sort_gpu_ms,raster_gpu_ms,total_gpu_ms,submit_wait_with_ui_ms\n";
        for(const auto& s:samples_) {const auto& m=s.metrics;
            out<<s.round+1<<','<<s.k<<','<<s.iteration<<','<<s.warmup<<','<<m.projected<<','
               <<m.width<<','<<m.height<<','<<m.prepareMs<<','<<m.uploadMs<<','<<m.cullMs<<','<<m.sortMs<<','<<m.rasterMs<<','<<m.gpuMs<<','<<m.submitWaitMs<<'\n';}
        return out.str();
    }
private:
    std::vector<uint32_t> budgets_;
    std::vector<PointBenchmarkSample> samples_;
    uint32_t stage_=0,iteration_=0;
    bool running_=false;
};

inline uint64_t BenchmarkUidHash(uint64_t uid) {
    uid+=0x9e3779b97f4a7c15ull;uid=(uid^(uid>>30))*0xbf58476d1ce4e5b9ull;
    uid=(uid^(uid>>27))*0x94d049bb133111ebull;return uid^(uid>>31);
}
inline uint64_t TimestampTicks(uint64_t start,uint64_t end,uint32_t bits) {
    if(!bits||bits>64)throw std::runtime_error("Vulkan queue has no valid timestamp bits");
    return bits==64?end-start:(end-start)&((uint64_t(1)<<bits)-1);
}
}  // namespace mycartgs
