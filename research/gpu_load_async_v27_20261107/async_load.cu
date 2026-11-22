#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "duty_budget.h"

using Clock = std::chrono::steady_clock;
constexpr int threadsPerBlock = 512;

struct BlockTiming {
    unsigned long long begin, end, workBegin, workEnd;
    unsigned int sm, padding;
};

static void checked(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}

static int64_t nanoseconds(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

static double seconds(Clock::time_point time) {
    return std::chrono::duration<double>(time.time_since_epoch()).count();
}

__device__ unsigned long long gpuTime() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ void armPulse(unsigned long long duration,unsigned long long* deadline) {
    if (threadIdx.x==0) *deadline=gpuTime()+duration;
}

__global__ __launch_bounds__(threadsPerBlock, 1)
void pulse(const unsigned long long* commonDeadline, float* output, BlockTiming* timing) {
    extern __shared__ unsigned char reserved[];
    if (threadIdx.x == 0) {
        unsigned int sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        timing[blockIdx.x].begin = gpuTime();
        timing[blockIdx.x].sm = sm;
        reserved[0] = static_cast<unsigned char>(blockIdx.x);
    }
    __syncthreads();
    if (threadIdx.x==0) timing[blockIdx.x].workBegin=gpuTime();
    const auto deadline = *commonDeadline;
    float a=.01f,b=.02f,c=.03f,d=.04f,e=.05f,f=.06f,g=.07f,h=.08f;
    while (gpuTime() < deadline) {
        #pragma unroll 8
        for (int iteration=0; iteration<8; ++iteration) {
            a=fmaf(a,.999999881f,.000000119f);
            b=fmaf(b,.999999881f,.000000120f);
            c=fmaf(c,.999999881f,.000000121f);
            d=fmaf(d,.999999881f,.000000122f);
            e=fmaf(e,.999999881f,.000000123f);
            f=fmaf(f,.999999881f,.000000124f);
            g=fmaf(g,.999999881f,.000000125f);
            h=fmaf(h,.999999881f,.000000126f);
        }
    }
    output[blockIdx.x*blockDim.x+threadIdx.x]=a+b+c+d+e+f+g+h+reserved[0];
    if (threadIdx.x==0) timing[blockIdx.x].workEnd=gpuTime();
    __syncthreads();
    if (threadIdx.x == 0) timing[blockIdx.x].end=gpuTime();
}

class Load {
    cudaDeviceProp device_{};
    cudaStream_t stream_{};
    cudaEvent_t begin_{},end_{};
    cudaGraph_t graph_{};
    cudaGraphExec_t plan_{};
    float* output_=nullptr;
    unsigned long long* deadline_=nullptr;
    BlockTiming *host_=nullptr,*gpu_=nullptr;
    int shared_=0,blocksPerSm_=0,totalBlocks_=0,priority_=0;
    unsigned long long cycle_;
    double duty_;
    int64_t initialDelay_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    std::ofstream pulses_,windows_;
    bool enabled_=false,paused_=true,stopping_=false,finished_=false;
    long generation_=0;
    int64_t completed_=0,charged_=0,resident_=0,tail_=0;
    Clock::time_point windowStart_;
    std::string error_,information_,summary_;

    int64_t activeNow() const {
        return completed_+nanoseconds(Clock::now()-windowStart_);
    }

    void work() {
        try {
            checked(cudaSetDevice(0));
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stopping_) {
                condition_.wait(lock,[&]{return !paused_ || stopping_;});
                if (stopping_) break;
                const auto startResident=resident_,startCharged=charged_;
                const auto generation=generation_;
                long count=0;
                while (enabled_ && !stopping_) {
                    if (duty_ == 0.) {
                        condition_.wait(lock,[&]{return !enabled_ || stopping_;});
                        continue;
                    }
                    const auto eligible=initialDelay_+requiredWindowNanoseconds(charged_,duty_);
                    const auto remaining=eligible-activeNow();
                    if (remaining > 0) {
                        condition_.wait_until(lock,Clock::now()+std::chrono::nanoseconds(remaining),
                                             [&]{return !enabled_ || stopping_;});
                        continue;
                    }
                    lock.unlock();
                    const auto submit=Clock::now();
                    checked(cudaEventRecord(begin_,stream_));
                    checked(cudaGraphLaunch(plan_,stream_));
                    checked(cudaEventRecord(end_,stream_));
                    checked(cudaEventSynchronize(end_));
                    const auto drained=Clock::now();
                    float eventMs=0.;
                    checked(cudaEventElapsedTime(&eventMs,begin_,end_));
                    auto first=host_[0].begin,last=host_[0].end;
                    std::map<unsigned int,int> sms;
                    unsigned long long startWait=0,workTime=0,endWait=0;
                    for (int index=0;index<totalBlocks_;++index) {
                        if (host_[index].end < host_[index].begin) throw std::runtime_error("Invalid GPU timestamp");
                        first=std::min(first,host_[index].begin);
                        last=std::max(last,host_[index].end);
                        startWait=std::max(startWait,host_[index].workBegin-host_[index].begin);
                        workTime=std::max(workTime,host_[index].workEnd-host_[index].workBegin);
                        endWait=std::max(endWait,host_[index].end-host_[index].workEnd);
                        ++sms[host_[index].sm];
                    }
                    int minimumBlocks=totalBlocks_,maximumBlocks=0;
                    for (const auto& item:sms) {
                        minimumBlocks=std::min(minimumBlocks,item.second);
                        maximumBlocks=std::max(maximumBlocks,item.second);
                    }
                    const int64_t resident=last-first;
                    const int64_t charged=std::max(resident,static_cast<int64_t>(std::ceil(eventMs*1000000.)));
                    lock.lock();
                    resident_+=resident;
                    charged_+=charged;
                    ++count;
                    pulses_<<generation<<','<<seconds(submit)<<','<<seconds(drained)<<','<<eventMs<<','
                           <<resident<<','<<charged<<','<<sms.size()<<','<<blocksPerSm_<<','
                           <<startWait<<','<<workTime<<','<<endWait<<','<<minimumBlocks<<','<<maximumBlocks<<'\n';
                }
                const auto finish=Clock::now();
                const auto elapsed=nanoseconds(finish-windowStart_);
                completed_+=elapsed;
                windows_<<generation<<','<<seconds(windowStart_)<<','<<seconds(finish)<<','<<elapsed<<','
                        <<resident_-startResident<<','<<charged_-startCharged<<','<<count<<'\n';
                paused_=true;
                condition_.notify_all();
            }
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_=error.what();
            paused_=true;
            condition_.notify_all();
        }
    }

public:
    Load(const char* filename,unsigned long long cycle,double duty,double phase)
      : cycle_(cycle),duty_(duty),initialDelay_(static_cast<int64_t>(phase*cycle)),
        pulses_(filename),windows_(std::string(filename)+".windows.csv") {
        validateDuty(duty);
        if (cycle<100000 || cycle>2000000 || !std::isfinite(phase) || phase<0. || phase>=1.) {
            throw std::invalid_argument("Cycle must be 0.1 to 2 ms and phase in [0,1)");
        }
        if (!pulses_ || !windows_) throw std::runtime_error("Cannot create load logs");
        checked(cudaGetDeviceProperties(&device_,0));
        if (!device_.cooperativeLaunch || !device_.canMapHostMemory) throw std::runtime_error("Unsupported device");
        int least,greatest;
        checked(cudaDeviceGetStreamPriorityRange(&least,&greatest));
        priority_=std::min(least,greatest+1);
        checked(cudaStreamCreateWithPriority(&stream_,cudaStreamNonBlocking,priority_));
        cudaFuncAttributes attributes;
        checked(cudaFuncGetAttributes(&attributes,pulse));
        const int desired=device_.maxThreadsPerMultiProcessor/threadsPerBlock;
        if (desired*threadsPerBlock!=device_.maxThreadsPerMultiProcessor) throw std::runtime_error("Unsupported thread capacity");
        shared_=std::min(device_.sharedMemPerBlockOptin,device_.sharedMemPerMultiprocessor/desired)-attributes.sharedSizeBytes;
        shared_=shared_/256*256;
        checked(cudaFuncSetAttribute(pulse,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_));
        checked(cudaFuncSetAttribute(pulse,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        do {
            checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm_,pulse,threadsPerBlock,shared_));
            if (blocksPerSm_>=desired) break;
            shared_-=256;
        } while (shared_>=0);
        if (blocksPerSm_!=desired) throw std::runtime_error("Cannot reserve all thread slots");
        totalBlocks_=device_.multiProcessorCount*blocksPerSm_*2;
        checked(cudaMalloc(&output_,totalBlocks_*threadsPerBlock*sizeof(float)));
        checked(cudaMalloc(&deadline_,sizeof(unsigned long long)));
        checked(cudaHostAlloc(&host_,totalBlocks_*sizeof(BlockTiming),cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(&gpu_,host_,0));
        checked(cudaEventCreate(&begin_));
        checked(cudaEventCreate(&end_));
        unsigned long long duration=static_cast<unsigned long long>(cycle_*duty_);
        void* arguments[]={&deadline_,&output_,&gpu_};
        checked(cudaStreamBeginCapture(stream_,cudaStreamCaptureModeThreadLocal));
        armPulse<<<1,32,0,stream_>>>(duration,deadline_);
        checked(cudaGetLastError());
        checked(cudaLaunchKernel(reinterpret_cast<void*>(pulse),dim3(totalBlocks_),
            dim3(threadsPerBlock),arguments,shared_,stream_));
        checked(cudaStreamEndCapture(stream_,&graph_));
        checked(cudaGraphInstantiate(&plan_,graph_,nullptr,nullptr,0));
        pulses_<<std::fixed<<std::setprecision(9)<<"generation,submit_monotonic_s,drained_monotonic_s,event_ms,resident_ns,charged_ns,unique_sms,blocks_each_sm,start_barrier_ns,work_ns,end_barrier_ns,minimum_blocks,maximum_blocks\n";
        windows_<<std::fixed<<std::setprecision(9)<<"generation,start_monotonic_s,end_monotonic_s,active_ns,resident_ns,charged_ns,pulses\n";
        std::ostringstream info;
        info<<"{\"sm_count\":"<<device_.multiProcessorCount<<",\"blocks_per_sm\":"<<blocksPerSm_
            <<",\"threads_per_sm\":"<<blocksPerSm_*threadsPerBlock<<",\"shared_per_block\":"<<shared_
            <<",\"stream_priority\":"<<priority_<<",\"highest_priority\":"<<greatest
            <<",\"cycle_ns\":"<<cycle_<<",\"target_duty\":"<<duty_<<",\"initial_delay_ns\":"<<initialDelay_
            <<",\"grid_barriers\":false,\"common_gpu_deadline\":true,\"pulse_graph\":true,\"grid_blocks\":"<<totalBlocks_<<'}';
        information_=info.str();
        thread_=std::thread(&Load::work,this);
    }

    const char* information() const {return information_.c_str();}

    long resume() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
        if (!paused_ || finished_) throw std::runtime_error("Invalid load resume");
        windowStart_=Clock::now();
        ++generation_;
        enabled_=true;
        paused_=false;
        condition_.notify_all();
        return generation_;
    }

    void pause() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (paused_) return;
        enabled_=false;
        condition_.notify_all();
        condition_.wait(lock,[&]{return paused_ || !error_.empty();});
        if (!error_.empty()) throw std::runtime_error(error_);
    }

    const char* finish() {
        pause();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
        if (finished_) return summary_.c_str();
        const auto remainder=remainingOffNanoseconds(charged_,completed_,duty_);
        const auto begin=Clock::now();
        lock.unlock();
        if (remainder) std::this_thread::sleep_for(std::chrono::nanoseconds(remainder));
        lock.lock();
        tail_=nanoseconds(Clock::now()-begin);
        const auto settled=completed_+tail_;
        if (settled<requiredWindowNanoseconds(charged_,duty_)) throw std::runtime_error("Duty cap violated");
        pulses_.flush();
        windows_.flush();
        std::ostringstream summary;
        summary<<std::setprecision(17)<<"{\"requests\":"<<generation_<<",\"active_ns\":"<<completed_
            <<",\"tail_ns\":"<<tail_<<",\"resident_ns\":"<<resident_<<",\"charged_ns\":"<<charged_
            <<",\"actual_duty\":"<<(settled?double(resident_)/settled:0.)
            <<",\"charged_duty\":"<<(settled?double(charged_)/settled:0.)
            <<",\"target_duty\":"<<duty_<<",\"phase_carried_across_frames\":true}";
        summary_=summary.str();
        finished_=true;
        return summary_.c_str();
    }

    ~Load() {
        try {finish();} catch (...) {}
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_=true;
            condition_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
        cudaStreamSynchronize(stream_);
        cudaGraphExecDestroy(plan_);
        cudaGraphDestroy(graph_);
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
        cudaFree(output_);
        cudaFree(deadline_);
        cudaFreeHost(host_);
        cudaStreamDestroy(stream_);
    }
};

static thread_local std::string errorMessage;
extern "C" {
void* phase_load_create(const char* filename,unsigned long long cycle,double duty,double phase) {
    try {return new Load(filename,cycle,duty,phase);}
    catch (const std::exception& error) {errorMessage=error.what();return nullptr;}
}
long phase_load_resume(void* object) {
    try {return static_cast<Load*>(object)->resume();}
    catch (const std::exception& error) {errorMessage=error.what();return -1;}
}
int phase_load_pause(void* object) {
    try {static_cast<Load*>(object)->pause();return 0;}
    catch (const std::exception& error) {errorMessage=error.what();return -1;}
}
const char* phase_load_finish(void* object) {
    try {return static_cast<Load*>(object)->finish();}
    catch (const std::exception& error) {errorMessage=error.what();return nullptr;}
}
const char* phase_load_information(void* object) {return static_cast<Load*>(object)->information();}
void phase_load_destroy(void* object) {delete static_cast<Load*>(object);}
const char* phase_load_error() {return errorMessage.c_str();}
}
