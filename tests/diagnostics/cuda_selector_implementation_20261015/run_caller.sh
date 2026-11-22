#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
PRODUCTION=/home/prmi/zwj/mycartgs
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
while [ ! -f "$D/integration_build.exit" ]; do sleep 2; done
test "$(cat "$D/integration_build.exit")" = 0
"$CMAKE" -S "$D" -B "$D/build-caller" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTorch_DIR=$PRODUCTION/third_party/libtorch-cu118/libtorch/share/cmake/Torch \
  -DOpenCV_DIR=$PRODUCTION/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4 \
  -DEigen3_DIR=$PRODUCTION/third_party/install/eigen-3.3.7/share/eigen3/cmake \
  -Djsoncpp_DIR=$PRODUCTION/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp \
  -DSELECTOR_INTEGRATION_VALIDATION=ON > "$D/caller_configure.log" 2>&1
"$CMAKE" --build "$D/build-caller" -j2 --target caller_replay > "$D/caller_build.log" 2>&1
echo 0 > "$D/caller_build.exit"
"$D/build-caller/caller_replay" "$D" benchmark > "$D/caller_benchmark.log" 2>&1
echo 0 > "$D/caller_benchmark.exit"
"$D/build-caller/caller_replay" "$D" replay 1000 > "$D/fixed_map_replay.log" 2>&1
echo 0 > "$D/fixed_map_replay.exit"
