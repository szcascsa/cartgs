#include <cuda_runtime.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using Clock = std::chrono::steady_clock;

static double seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void loadPulse(unsigned long long cycles, float* output) {
    const auto start = clock64();
    float value = 0.01f * (threadIdx.x + 1);
    while (clock64() - start < cycles) {
        #pragma unroll 8
        for (int i = 0; i < 32; ++i) value = fmaf(value, 1.000000119f, 0.000000119f);
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

class RenderLoad {
    cudaStream_t stream_{};
    cudaEvent_t begin_{}, end_{};
    float* output_ = nullptr;
    cudaDeviceProp properties_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    std::ofstream log_;
    bool enabled_ = false, paused_ = true, stopping_ = false;
    long generation_ = 0, armed_ = 0;
    int stage_ = -1;
    double duty_ = 0.;
    std::string error_;

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
                if (duty == 0.) {
                    armed_ = generation;
                    condition_.notify_all();
                    condition_.wait(lock, [&] { return !enabled_ || stopping_; });
                } else {
                    while (enabled_ && !stopping_) {
                        lock.unlock();
                        const auto cycle = Clock::now();
                        const double start = seconds();
                        const int pulses = static_cast<int>(std::ceil(duty * 5.));
                        const auto cycles = static_cast<unsigned long long>(
                            properties_.clockRate * duty / pulses);
                        checked(cudaEventRecord(begin_, stream_));
                        for (int i = 0; i < pulses; ++i) {
                            loadPulse<<<properties_.multiProcessorCount * 2, 256, 0, stream_>>>(cycles, output_);
                        }
                        checked(cudaGetLastError());
                        checked(cudaEventRecord(end_, stream_));
                        lock.lock();
                        armed_ = generation;
                        condition_.notify_all();
                        lock.unlock();
                        checked(cudaEventSynchronize(end_));
                        float gpu_ms = 0.;
                        checked(cudaEventElapsedTime(&gpu_ms, begin_, end_));
                        const double finish = seconds();
                        log_ << generation << ',' << stage << ',' << duty << ',' << start << ','
                             << finish << ',' << gpu_ms << ',' << pulses << '\n';
                        lock.lock();
                        condition_.wait_until(lock, cycle + std::chrono::milliseconds(1),
                            [&] { return !enabled_ || stopping_; });
                    }
                }
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
    int leastPriority = 0, greatestPriority = 0;

    explicit RenderLoad(const char* path) : log_(path) {
        if (!log_) throw std::runtime_error("Cannot open load audit log");
        checked(cudaGetDeviceProperties(&properties_, 0));
        checked(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        checked(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, leastPriority));
        checked(cudaMalloc(&output_, properties_.multiProcessorCount * 2 * 256 * sizeof(float)));
        checked(cudaEventCreate(&begin_));
        checked(cudaEventCreate(&end_));
        log_ << std::fixed << std::setprecision(9);
        log_ << "generation,stage,target_duty,submit_monotonic_s,drained_monotonic_s,event_ms,pulses\n";
        thread_ = std::thread(&RenderLoad::work, this);
    }

    long resume(int stage, double duty) {
        if (!std::isfinite(duty) || duty < 0 || duty > .8) throw std::runtime_error("Invalid duty");
        std::unique_lock<std::mutex> lock(mutex_);
        if (!error_.empty()) throw std::runtime_error(error_);
        if (!paused_) throw std::runtime_error("Load must be drained before resuming");
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
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
        cudaStreamDestroy(stream_);
    }
};

static thread_local std::string lastError;

extern "C" {
void* render_load_create(const char* path) {
    try { return new RenderLoad(path); }
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
void render_load_destroy(void* object) { delete static_cast<RenderLoad*>(object); }
const char* render_load_error() { return lastError.c_str(); }
}
