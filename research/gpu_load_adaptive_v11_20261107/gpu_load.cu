#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

using Clock = std::chrono::steady_clock;

__global__ void load_pulse(unsigned long long cycles, float* output) {
    const auto start = clock64();
    float value = 0.01f * (threadIdx.x + 1);
    while (clock64() - start < cycles) {
        #pragma unroll 8
        for (int i = 0; i < 32; ++i) value = fmaf(value, 1.000000119f, 0.000000119f);
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

std::string argument(int argc, char** argv, const std::string& name) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    throw std::runtime_error("Missing argument " + name);
}

int main(int argc, char** argv) {
    try {
        const auto ready = argument(argc, argv, "--ready-file");
        const auto command = argument(argc, argv, "--command-file");
        const auto stop = argument(argc, argv, "--stop-file");
        const auto log_path = argument(argc, argv, "--log");
        const int parent_pid = std::stoi(argument(argc, argv, "--parent-pid"));
        cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties, 0));
        const int blocks = properties.multiProcessorCount * 2;
        float* output = nullptr;
        checked(cudaMalloc(&output, blocks * 256 * sizeof(float)));
        cudaEvent_t begin, end;
        checked(cudaEventCreate(&begin));
        checked(cudaEventCreate(&end));
        std::ofstream log(log_path);
        if (!log) throw std::runtime_error("Cannot create load log");
        log << std::fixed << std::setprecision(9);
        log << "monotonic_s,stage,target_duty,pulses,event_ms,wall_ms\n";
        std::ofstream(ready) << "ready\n";
        const auto epoch = Clock::now();
        auto next_log = epoch;
        int stage = -1;
        double duty = 0.;
        while (!std::filesystem::exists(stop)) {
            if (getppid() != parent_pid || Clock::now() - epoch > std::chrono::seconds(60)) break;
            if (std::filesystem::exists(command)) {
                std::ifstream input(command);
                int new_stage;
                double new_duty;
                if (input >> new_stage >> new_duty) {
                    if (new_duty < 0 || new_duty > .8) throw std::runtime_error("Invalid duty");
                    stage = new_stage;
                    duty = new_duty;
                }
            }
            const auto cycle = Clock::now();
            const double on_us = 1000. * duty;
            const int pulses = static_cast<int>(std::ceil(on_us / 200.));
            float elapsed_ms = 0.;
            if (pulses) {
                const auto cycles = static_cast<unsigned long long>(properties.clockRate * on_us / (1000. * pulses));
                checked(cudaEventRecord(begin));
                for (int i = 0; i < pulses; ++i) load_pulse<<<blocks, 256>>>(cycles, output);
                checked(cudaGetLastError());
                checked(cudaEventRecord(end));
                checked(cudaEventSynchronize(end));
                checked(cudaEventElapsedTime(&elapsed_ms, begin, end));
            }
            const auto complete = Clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(complete - cycle).count();
            if (complete >= next_log) {
                log << std::chrono::duration<double>(complete.time_since_epoch()).count() << ','
                    << stage << ',' << duty << ',' << pulses << ',' << elapsed_ms << ',' << wall_ms << '\n' << std::flush;
                next_log = complete + std::chrono::milliseconds(50);
            }
            std::this_thread::sleep_until(cycle + std::chrono::microseconds(1000));
        }
        checked(cudaFree(output));
        checked(cudaEventDestroy(begin));
        checked(cudaEventDestroy(end));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GPU_LOAD_ERROR %s\n", error.what());
        return 1;
    }
}
