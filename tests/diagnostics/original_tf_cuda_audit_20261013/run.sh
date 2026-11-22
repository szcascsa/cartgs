#!/usr/bin/env bash
set -euo pipefail
cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_tf_cuda_audit_20261013
trap 'rc=$?; printf "%s\n" "$rc" > run.exit' EXIT
PY=/home/prmi/anaconda3/envs/zwj/bin/python
export PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
"$PY" -u prepare.py > prepare.log 2>&1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch > build.log 2>&1
cmake --build build -j2 >> build.log 2>&1
./build/native_benchmark /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply . input.bin > native.log 2>&1
./build/native_benchmark /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply . input.bin isolated > isolated.log 2>&1
"$PY" -u profile.py > profile.log 2>&1
"$PY" reparse_profile.py > reparse.log 2>&1
"$PY" collect_log_shapes.py > logged_shapes.log 2>&1
"$PY" autograd_graph.py > autograd_graph.log 2>&1
