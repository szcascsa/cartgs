#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
trap 'rc=$?; printf "%s\n" "$rc" > tests/diagnostics/cuda_tf_implementation_20261013/online_build.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S . -B build-cuda118-sm86-gcc9
"$CMAKE" --build build-cuda118-sm86-gcc9 --target tum_rgbd_cuda_tf_online -j2
