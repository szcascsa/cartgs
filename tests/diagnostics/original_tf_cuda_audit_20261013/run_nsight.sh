#!/usr/bin/env bash
set -euo pipefail
cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_tf_cuda_audit_20261013
trap 'rc=$?; printf "%s\n" "$rc" > nsight.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
nsys --version > nsight_version.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch > nsight_build.log 2>&1
cmake --build build -j2 >> nsight_build.log 2>&1
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true --output=native_nsys ./build/native_benchmark /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply . input.bin profile
nsys export --type=sqlite --force-overwrite=true --output=native_nsys.sqlite native_nsys.nsys-rep
/home/prmi/anaconda3/envs/zwj/bin/python parse_nsight.py > nsight_summary.log
