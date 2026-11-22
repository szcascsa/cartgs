#include "point_benchmark.h"
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using namespace mycartgs;
void Check(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
int main() {
    PointBenchmark b;b.Start(43197);
    std::vector<uint32_t> cases;
    while(b.Running()) {
        const auto stage=b.Stage(),k=b.K();cases.push_back(k);
        while(b.Running()&&b.Stage()==stage) {
            PointBenchmarkMetrics m;m.projected=k/3;m.width=1200;m.height=680;
            m.gpuMs=b.Iteration()<3?1000.:double(b.Iteration()-2);
            b.Record(m);
        }
    }
    Check(cases.size()==18&&cases.front()==0&&cases[8]==43197&&cases[9]==43197&&cases.back()==0,"Counterbalanced K order changed");
    Check(b.Samples().size()==324,"Sample capacity or progress changed");
    for(const auto& s:b.Summaries()) {
        Check(s.samples==30&&s.medianMs==8&&s.p95Ms==15,"Warmup/median/p95 calculation failed");
        Check(s.projected==s.k/3,"Projected count lost");
    }
    Check(b.Csv().find("prepare_once_ms,upload_ms,raster_gpu_ms,submit_wait_with_ui_ms")!=std::string::npos,"CSV timing scope lost");
    b.Start(1500);std::set<uint32_t> counts;
    while(b.Running()){counts.insert(b.K());b.Record({});}
    Check(counts==std::set<uint32_t>{0,1000,1500},"Small-map K clamp/dedup failed");
    b.Start(0);while(b.Running())b.Record({});
    Check(b.Summaries().size()==1&&b.Summaries()[0].samples==30,"Empty map case failed");
    b.Start(1000);b.Record({});b.Cancel();Check(!b.Running()&&b.Summaries().empty(),"Cancellation counts warmup as measured");
    b.Start(2000);Check(b.Samples().empty()&&b.K()==0,"Rerun retained previous samples");
    bool rejected=false;try{PointBenchmarkMetrics m;m.gpuMs=NAN;b.Record(m);}catch(const std::runtime_error&){rejected=true;}
    Check(rejected,"Nonfinite timing accepted");
    Check(TimestampTicks(0xfffffff0,0x10,32)==32,"Timestamp wrap failed");
    Check(TimestampTicks((uint64_t(1)<<48)-2,3,48)==5,"Quest 48-bit timestamp wrap failed");
    Check(TimestampTicks(std::numeric_limits<uint64_t>::max()-2,2,64)==5,"64-bit timestamp wrap failed");
    std::set<uint64_t> hashes;for(uint64_t uid=0;uid<43197;++uid)hashes.insert(BenchmarkUidHash(uid));
    Check(hashes.size()==43197,"UID ranking unexpectedly duplicated");
    std::cout<<"point_benchmark_test: PASS (bounded schedule, warmup, reverse order, median/p95, empty, cancel, rerun, timestamp wrap)\n";
}
