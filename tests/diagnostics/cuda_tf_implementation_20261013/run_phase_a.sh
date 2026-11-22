#!/usr/bin/env bash
set -euo pipefail
cd /home/prmi/zwj/mycartgs/tests/diagnostics/cuda_tf_implementation_20261013
trap 'rc=$?; printf "%s\n" "$rc" > phase_a.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch > phase_a_build.log 2>&1
"$CMAKE" --build build -j2 >> phase_a_build.log 2>&1
./build/cuda_transform_field_test . /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply > phase_a_test.log 2>&1
./build/phase1_benchmark /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply ../original_tf_cuda_audit_20261013/input.bin phase_1_benchmark.csv > phase_1_benchmark.log 2>&1
