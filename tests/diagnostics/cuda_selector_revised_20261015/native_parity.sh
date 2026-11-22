#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
PRODUCTION=/home/prmi/zwj/mycartgs
D=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
"$CMAKE" -S "$D" -B "$D/build" -DSELECTOR_CUDA_CANDIDATES=ON -DSELECTOR_INTEGRATION_VALIDATION=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 \
  -DTorch_DIR=$PRODUCTION/third_party/libtorch-cu118/libtorch/share/cmake/Torch \
  -DOpenCV_DIR=$PRODUCTION/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4 \
  -DEigen3_DIR=$PRODUCTION/third_party/install/eigen-3.3.7/share/eigen3/cmake \
  -Djsoncpp_DIR=$PRODUCTION/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp > "$D/configure.log" 2>&1
"$CMAKE" --build "$D/build" -j2 > "$D/build.log" 2>&1
while [ ! -f "$D/selection.exit" ]; do sleep 2; done
test "$(cat "$D/selection.exit")" = 0
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
"$D/build/selector_parity" --candidate cublas_strict --device cuda --rows 257 --steps 1000 \
  --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out "$D/native_1000.csv" > "$D/native_1000.log" 2>&1
"$D/build/selector_parity" --candidate cublas_strict --device cuda --rows 219969 --steps 0 \
  --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out "$D/native_219969.csv" > "$D/native_219969.log" 2>&1
/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest --test-dir "$ROOT/build-integration" --output-on-failure \
  -R '(change_distill_selector_cuda|drift_stability_selector_cuda|joint_replay_selector_cuda|historical_gradient_selector_cuda|^gradient_constraint$)' > "$D/integration.log" 2>&1
echo 0 > "$D/native_parity.exit"
PREVIOUS=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
mkdir -p "$D/replay_room/current_checkpoint"
cp "$PREVIOUS"/current_*.bin "$PREVIOUS/current_spec.txt" "$D/replay_room/"
cp "$PREVIOUS"/current_checkpoint/* "$D/replay_room/current_checkpoint/"
cp "$PREVIOUS/fixed_map_metrics.py" "$D/replay_room/"
"$D/build/caller_replay" "$D/replay_room" joint 1000 > "$D/joint_1000.log" 2>&1
/home/prmi/anaconda3/envs/zwj/bin/python "$D/replay_room/fixed_map_metrics.py" joint_fixed_map > "$D/joint_1000_metrics.log" 2>&1
echo 0 > "$D/training_parity.exit"
