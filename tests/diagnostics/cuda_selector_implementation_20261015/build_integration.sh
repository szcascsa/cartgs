#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
PRODUCTION=/home/prmi/zwj/mycartgs
cd "$ROOT"
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
CTEST=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
test ! -e third_party || test -L third_party
test -e third_party || ln -s "$PRODUCTION/third_party" third_party
"$CMAKE" -S . -B build-integration -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTorch_DIR=$PRODUCTION/third_party/libtorch-cu118/libtorch/share/cmake/Torch \
  -DOpenCV_DIR=$PRODUCTION/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4 \
  -DEigen3_DIR=$PRODUCTION/third_party/install/eigen-3.3.7/share/eigen3/cmake \
  -Djsoncpp_DIR=$PRODUCTION/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp \
  -DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON -DCARTGS_BUILD_TESTS=ON > "$D/integration_configure.log" 2>&1
"$CMAKE" --build build-integration -j2 --target tum_rgbd tum_rgbd_cuda_tf_online change_distill_selector_cuda_test \
  drift_stability_selector_cuda_test joint_replay_selector_cuda_test historical_gradient_selector_cuda_test \
  gradient_constraint_test > "$D/integration_build.log" 2>&1
echo 0 > "$D/integration_build.exit"
"$CTEST" --test-dir build-integration --output-on-failure \
  -R '(change_distill_selector_cuda|drift_stability_selector_cuda|joint_replay_selector_cuda|historical_gradient_selector_cuda|^gradient_constraint$)' \
  > "$D/integration_tests.log" 2>&1
echo 0 > "$D/integration_tests.exit"
