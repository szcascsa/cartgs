#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
TF_DIAGNOSTICS=tests/diagnostics/cuda_tf_implementation_20261013
trap 'rc=$?; printf "%s\n" "$rc" > "$TF_DIAGNOSTICS/main_build.exit"' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:/home/prmi/zwj/mycartgs/lib:/home/prmi/zwj/mycartgs/third_party/ORB-SLAM3/lib
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S . -B build-cuda118-sm86-gcc9 -DCARTGS_BUILD_TESTS=ON
"$CMAKE" --build build-cuda118-sm86-gcc9 -j2
/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest --test-dir build-cuda118-sm86-gcc9 --output-on-failure -j1
