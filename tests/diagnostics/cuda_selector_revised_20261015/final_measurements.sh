#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
PRODUCTION=/home/prmi/zwj/mycartgs
D=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/final_measurements.exit"' EXIT
while [ ! -f "$D/final_native_build.exit" ]; do sleep 2; done
test "$(cat "$D/final_native_build.exit")" = 0
"$CMAKE" --build "$D/build" -j2 > "$D/final_diagnostic_build.log" 2>&1
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
mkdir "$D/final_measurement"
cd "$D/final_measurement"
"$D/build/selector_parity" --candidate cublas_strict --device cuda --rows 257 --steps 1000 \
 --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out native_1000.csv > native_1000.log 2>&1
"$D/build/selector_parity" --candidate cublas_strict --device cuda --rows 219969 --steps 0 \
 --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out native_219969.csv > native_219969.log 2>&1
"$D/build/optimizer_candidate" . > optimizer.log 2>&1
"$D/build/guard_transaction" guard.csv > guard.log 2>&1
"$D/build/caller_replay" "$D/replay_room" joint 1000 > joint_1000.log 2>&1
/home/prmi/anaconda3/envs/zwj/bin/python "$D/replay_room/fixed_map_metrics.py" joint_fixed_map > joint_quality.log 2>&1
for backend in legacy cublas_strict; do
 "$D/build/network_benchmark" "$CHECKPOINT" "$FIXTURE" "network_$backend.csv" "$backend" > "network_$backend.log" 2>&1
 "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o "mlp_$backend" "$D/build/network_benchmark" "$CHECKPOINT" "$FIXTURE" "profile_$backend.csv" "$backend" profile > "nsys_$backend.log" 2>&1
done
"$D/build/caller_replay" "$D/replay_room" benchmark > caller.log 2>&1
"$D/build/caller_replay" "$D/replay_room" grid "$FIXTURE" "$CHECKPOINT" > caller_grid.log 2>&1
