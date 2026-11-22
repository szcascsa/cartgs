#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "duty_budget.h"

using Clock = std::chrono::steady_clock;
constexpr int threadsPerBlock = 512;
constexpr unsigned long long nominalCycleNanoseconds = 20000000;

struct BlockTiming {
    unsigned long long begin, end;
    unsigned int sm, padding;
};

static double seconds(Clock::time_point time) {
    return std::chrono::duration<double>(time.time_since_epoch()).count();
}

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__device__ unsigned long long globalNanoseconds() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__global__ __launch_bounds__(threadsPerBlock, 1)
void saturatedPulse(unsigned long long duration, float* output, BlockTiming* timing) {
    extern __shared__ unsigned char reserved[];
    if (threadIdx.x == 0) {
        unsigned int sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        timing[blockIdx.x].begin = globalNanoseconds();
        timing[blockIdx.x].sm = sm;
        reserved[0] = static_cast<unsigned char>(blockIdx.x);
    }
    cooperative_groups::this_grid().sync();
    const auto deadline = globalNanoseconds() + duration;
    float a = .01f, b = .02f, c = .03f, d = .04f;
    float e = .05f, f = .06f, g = .07f, h = .08f;
    while (globalNanoseconds() < deadline) {
        #pragma unroll 8
        for (int iteration = 0; iteration < 8; ++iteration) {
            a = fmaf(a, .999999881f, .000000119f);
            b = fmaf(b, .999999881f, .000000120f);
            c = fmaf(c, .999999881f, .000000121f);
            d = fmaf(d, .999999881f, .000000122f);
            e = fmaf(e, .999999881f, .000000123f);
            f = fmaf(f, .999999881f, .000000124f);
            g = fmaf(g, .999999881f, .000000125f);
            h = fmaf(h, .999999881f, .000000126f);
        }
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = a + b + c + d + e + f + g + h + reserved[0];
    cooperative_groups::this_grid().sync();
    if (threadIdx.x == 0) timing[blockIdx.x].end = globalNanoseconds();
}

class RenderLoad {
    cudaDeviceProp device_{};
    cudaStream_t stream_{};
    cudaEvent_t begin_{}, end_{};
    float* output_ = nullptr;
    BlockTiming *hostTiming_ = nullptr, *deviceTiming_ = nullptr;
    int sharedBytes_ = 0, residentBlocks_ = 0, loadPriority_ = 0, registers_ = 0;
    int blockCount_ = 0;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    std::ofstream pulses_, phases_;
    bool enabled_ = false, paused_ = true, stopping_ = false;
    long generation_ = 0, armed_ = 0;
    int stage_ = -1;
    double duty_ = 0.;
    Clock::time_point phaseStart_;
    std::string error_, information_;

    void work() {
        try {
            checked(cudaSetDevice(0));
            setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
            std::unique_lock<std::mutex> lock(mutex_);
            while (!stopping_) {
                condition_.wait(lock, [&] { return enabled_ || stopping_; });
                if (stopping_) break;
                const long generation = generation_;
                const int stage = stage_;
                const double duty = duty_;
                const auto phaseStart = phaseStart_;
                int64_t charged = 0, measured = 0;
                long count = 0;
                if (duty == 0.) {
                    armed_ = generation;
                    condition_.notify_all();
                    condition_.wait(lock, [&] { return !enabled_ || stopping_; });
                } else {
                    while (enabled_ && !stopping_) {
                        lock.unlock();
                        const auto submit = Clock::now();
                        unsigned long long duration = static_cast<unsigned long long>(nominalCycleNanoseconds * duty);
                        void* arguments[] = {&duration, &output_, &deviceTiming_};
                        checked(cudaEventRecord(begin_, stream_));
                        checked(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(saturatedPulse),
                            dim3(blockCount_), dim3(threadsPerBlock), arguments, sharedBytes_, stream_));
                        checked(cudaEventRecord(end_, stream_));
                        lock.lock();
                        armed_ = generation;
                        condition_.notify_all();
                        lock.unlock();
                        checked(cudaEventSynchronize(end_));
                        const auto drained = Clock::now();
                        float eventMilliseconds = 0.;
                        checked(cudaEventElapsedTime(&eventMilliseconds, begin_, end_));
                        unsigned long long first = hostTiming_[0].begin, last = hostTiming_[0].end;
                        long double smNanoseconds = 0.;
                        std::map<unsigned int, int> usedSms;
                        for (int block = 0; block < blockCount_; ++block) {
                            const auto& record = hostTiming_[block];
                            if (record.end < record.begin) throw std::runtime_error("Invalid GPU timestamp");
                            first = std::min(first, record.begin);
                            last = std::max(last, record.end);
                            smNanoseconds += record.end - record.begin;
                            ++usedSms[record.sm];
                        }
                        if (usedSms.size() != static_cast<size_t>(device_.multiProcessorCount)) {
                            throw std::runtime_error("Cooperative pulse did not cover every SM");
                        }
                        for (const auto& item : usedSms) {
                            if (item.second != residentBlocks_) {
                                throw std::runtime_error("Pulse did not reserve every resident thread slot");
                            }
                        }
                        const int64_t residentNanoseconds = last - first;
                        const int64_t chargedNanoseconds = std::max(residentNanoseconds,
                            static_cast<int64_t>(std::ceil(eventMilliseconds * 1000000.)));
                        charged += chargedNanoseconds;
                        measured += residentNanoseconds;
                        ++count;
                        pulses_ << generation << ',' << stage << ',' << duty << ',' << seconds(submit) << ','
                            << seconds(drained) << ',' << eventMilliseconds << ',' << residentNanoseconds << ','
                            << static_cast<double>(smNanoseconds / blockCount_) << ','
                            << chargedNanoseconds << ',' << usedSms.size() << ',' << residentBlocks_ << '\n';
                        lock.lock();
                        const auto deadline = phaseStart + std::chrono::nanoseconds(requiredWindowNanoseconds(charged, duty));
                        // 停止当前窗口也要结清OFF时间，随后才允许下一次准备。
                        condition_.wait_until(lock, deadline, [] { return false; });
                    }
                }
                const auto finish = Clock::now();
                const int64_t elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - phaseStart).count();
                if (charged > 0 && elapsed < requiredWindowNanoseconds(charged, duty)) {
                    throw std::runtime_error("Unpaid OFF time at the preparation boundary");
                }
                phases_ << generation << ',' << stage << ',' << duty << ',' << seconds(phaseStart) << ','
                    << seconds(finish) << ',' << count << ',' << measured << ',' << charged << ',' << elapsed << ','
                    << (elapsed ? double(measured) / elapsed : 0.) << ','
                    << (elapsed ? double(charged) / elapsed : 0.) << '\n' << std::flush;
                pulses_.flush();
                paused_ = true;
                condition_.notify_all();
            }
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            error_ = error.what();
            paused_ = true;
            condition_.notify_all();
        }
    }

public:
    explicit RenderLoad(const char* filename) : pulses_(filename), phases_(std::string(filename) + ".phases.csv") {
        if (!pulses_ || !phases_) throw std::runtime_error("Cannot create duty audit logs");
        checked(cudaGetDeviceProperties(&device_, 0));
        if (!device_.cooperativeLaunch || !device_.canMapHostMemory) {
            throw std::runtime_error("Cooperative launch and mapped host timestamps are required");
        }
        int least, greatest;
        checked(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        loadPriority_ = std::min(least, greatest + 1);
        checked(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, loadPriority_));
        cudaFuncAttributes attributes;
        checked(cudaFuncGetAttributes(&attributes, saturatedPulse));
        registers_ = attributes.numRegs;
        const int desiredBlocks = device_.maxThreadsPerMultiProcessor / threadsPerBlock;
        if (desiredBlocks * threadsPerBlock != device_.maxThreadsPerMultiProcessor) {
            throw std::runtime_error("Device thread capacity must be divisible by the pulse block size");
        }
        sharedBytes_ = std::min(device_.sharedMemPerBlockOptin,
            device_.sharedMemPerMultiprocessor / desiredBlocks) - attributes.sharedSizeBytes;
        sharedBytes_ = sharedBytes_ / 256 * 256;
        checked(cudaFuncSetAttribute(saturatedPulse, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedBytes_));
        checked(cudaFuncSetAttribute(saturatedPulse, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        do {
            checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentBlocks_, saturatedPulse,
                threadsPerBlock, sharedBytes_));
            if (residentBlocks_ >= desiredBlocks) break;
            sharedBytes_ -= 256;
        } while (sharedBytes_ >= 0);
        if (residentBlocks_ != desiredBlocks) {
            throw std::runtime_error("Cannot reserve every resident thread slot");
        }
        blockCount_ = device_.multiProcessorCount * residentBlocks_;
        checked(cudaMalloc(&output_, blockCount_ * threadsPerBlock * sizeof(float)));
        checked(cudaHostAlloc(&hostTiming_, blockCount_ * sizeof(BlockTiming), cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(&deviceTiming_, hostTiming_, 0));
        checked(cudaEventCreate(&begin_));
        checked(cudaEventCreate(&end_));
        pulses_ << std::fixed << std::setprecision(9)
            << "generation,stage,target_duty,submit_monotonic_s,drained_monotonic_s,event_ms,resident_envelope_ns,mean_block_resident_ns,charged_ns,unique_sms,blocks_each_sm\n";
        phases_ << std::fixed << std::setprecision(9)
            << "generation,stage,target_duty,start_monotonic_s,end_monotonic_s,pulses,resident_envelope_ns,charged_ns,window_ns,measured_resident_duty,charged_duty\n";
        std::ostringstream info;
        info << "{\"sm_count\":" << device_.multiProcessorCount << ",\"threads_per_block\":" << threadsPerBlock
            << ",\"shared_bytes_per_block\":" << sharedBytes_ << ",\"shared_bytes_per_sm\":" << device_.sharedMemPerMultiprocessor
            << ",\"resident_blocks_per_sm\":" << residentBlocks_ << ",\"registers_per_thread\":" << registers_
            << ",\"resident_threads_per_sm\":" << residentBlocks_ * threadsPerBlock
            << ",\"maximum_threads_per_sm\":" << device_.maxThreadsPerMultiProcessor
            << ",\"total_blocks\":" << blockCount_
            << ",\"load_stream_priority\":" << loadPriority_ << ",\"preparation_highest_priority\":" << greatest
            << ",\"nominal_cycle_ns\":" << nominalCycleNanoseconds
            << ",\"maximum_on_ns\":" << static_cast<unsigned long long>(nominalCycleNanoseconds * maximumDuty)
            << ",\"maximum_duty\":0.8}";
        information_ = info.str();
        thread_ = std::thread(&RenderLoad::work, this);
    }

    const char* information() const { return information_.c_str(); }

    long resume(int stage, double duty) {
        validateDuty(duty);
        std::unique_lock<std::mutex> lock(mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
        if (!paused_) throw std::runtime_error("Drain load before the next preparation");
        phaseStart_ = Clock::now();
        stage_ = stage;
        duty_ = duty;
        enabled_ = true;
        paused_ = false;
        ++generation_;
        condition_.notify_all();
        condition_.wait(lock, [&] { return armed_ == generation_ || !error_.empty(); });
        if (!error_.empty()) throw std::runtime_error(error_);
        return generation_;
    }

    void pause() {
        std::unique_lock<std::mutex> lock(mutex_);
        enabled_ = false;
        condition_.notify_all();
        condition_.wait(lock, [&] { return paused_ || !error_.empty(); });
        if (!error_.empty()) throw std::runtime_error(error_);
    }

    ~RenderLoad() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            enabled_ = false;
            stopping_ = true;
            condition_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
        cudaStreamSynchronize(stream_);
        cudaFree(output_);
        cudaFreeHost(hostTiming_);
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
        cudaStreamDestroy(stream_);
    }
};

static thread_local std::string lastError;

extern "C" {
void* render_load_create(const char* filename) {
    try { return new RenderLoad(filename); }
    catch (const std::exception& error) { lastError = error.what(); return nullptr; }
}
long render_load_resume(void* object, int stage, double duty) {
    try { return static_cast<RenderLoad*>(object)->resume(stage, duty); }
    catch (const std::exception& error) { lastError = error.what(); return -1; }
}
int render_load_pause(void* object) {
    try { static_cast<RenderLoad*>(object)->pause(); return 0; }
    catch (const std::exception& error) { lastError = error.what(); return -1; }
}
const char* render_load_information(void* object) { return static_cast<RenderLoad*>(object)->information(); }
void render_load_destroy(void* object) { delete static_cast<RenderLoad*>(object); }
const char* render_load_error() { return lastError.c_str(); }
}
