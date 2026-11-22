#!/usr/bin/env bash
set -euo pipefail
cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_selector_cuda_audit_20261013
trap 'rc=$?; printf "%s\n" "$rc" > final.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch -DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4 -DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/install/eigen-3.3.7/share/eigen3/cmake > final_build.log 2>&1
cmake --build build -j2 >> final_build.log 2>&1
./build/gradient_paths . > gradient_paths.log 2>&1
./build/isolated_costs . > isolated_costs.log 2>&1
./build/native_benchmark . operators > operators.log 2>&1
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o native_nsys ./build/native_benchmark . profile > nsight.log 2>&1
nsys export --type sqlite --force-overwrite=true --output native_nsys.sqlite native_nsys.nsys-rep >> nsight.log 2>&1
