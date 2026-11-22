#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'rc=$?; printf "%s\n" "$rc" > phase_3.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S . -B build
"$CMAKE" --build build --target cuda_tf_regularization_test training_benchmark -j2
CHECKPOINT=/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply
build/cuda_tf_regularization_test . "$CHECKPOINT"
build/training_benchmark "$CHECKPOINT" ../original_tf_cuda_audit_20261013/input.bin phase_3
