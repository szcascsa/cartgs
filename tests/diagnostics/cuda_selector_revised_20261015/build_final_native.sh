#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
CTEST=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest
export LD_LIBRARY_PATH=$ROOT/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/final_native_build.exit"' EXIT
cd "$ROOT"
"$CMAKE" -S . -B build-integration > "$D/final_native_configure.log" 2>&1
"$CMAKE" --build build-integration -j2 --target tum_rgbd_selector_validation tum_rgbd_cuda_tf_online \
 change_distill_selector_cuda_test drift_stability_selector_cuda_test joint_replay_selector_cuda_test \
 historical_gradient_selector_cuda_test gradient_constraint_test > "$D/final_native_build.log" 2>&1
"$CTEST" --test-dir build-integration --output-on-failure \
 -R '(change_distill_selector_cuda|drift_stability_selector_cuda|joint_replay_selector_cuda|historical_gradient_selector_cuda|^gradient_constraint$)' > "$D/final_integration.log" 2>&1
