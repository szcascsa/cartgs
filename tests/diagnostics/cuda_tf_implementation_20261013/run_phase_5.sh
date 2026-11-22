#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'rc=$?; printf "%s\n" "$rc" > phase_5.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" --build build --target training_benchmark -j2
build/training_benchmark /home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply ../original_tf_cuda_audit_20261013/input.bin phase_5 cuda_optimizer
