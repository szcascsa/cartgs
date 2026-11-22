#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
PYTHON=/home/prmi/anaconda3/envs/zwj/bin/python
export LD_LIBRARY_PATH=$ROOT/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/fixed_map.exit"' EXIT
while [ ! -f "$D/final_measurements.exit" ]; do sleep 2; done
test "$(cat "$D/final_measurements.exit")" = 0
"$CMAKE" -S "$D" -B "$D/build" > "$D/fixed_map_configure.log" 2>&1
"$CMAKE" --build "$D/build" -j2 --target fixed_map > "$D/fixed_map_build.log" 2>&1
for scene in room desk xyz long; do
 "$PYTHON" "$D/prepare_fixed_map.py" "$scene" > "$D/fixed_map_prepare_$scene.log" 2>&1
 fixture=$D/fixed_map/$scene
 for pair in 'legacy legacy' 'legacy repeat' 'cuda cuda'; do
  read -r backend label <<< "$pair"
  "$D/build/fixed_map" "$fixture" "$backend" "$label" 1000 > "$fixture/$label.log" 2>&1
 done
 "$PYTHON" "$D/fixed_map_metrics.py" "$fixture" > "$fixture/quality.log" 2>&1
 # Keep the complete reference trace compressed after all independent comparisons.
 gzip "$fixture/independent_reference.bin"
done
