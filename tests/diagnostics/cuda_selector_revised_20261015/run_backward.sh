#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
PRODUCTION=/home/prmi/zwj/mycartgs
trap 'echo $? > "$D/backward.exit"' EXIT
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S "$D" -B "$D/build" > "$D/backward_configure.log" 2>&1
"$CMAKE" --build "$D/build" --target backward_benchmark -j2 > "$D/backward_build.log" 2>&1
cd "$D"
"$D/build/backward_benchmark" \
  "$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt" \
  "$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013" "$D/backward_candidates.csv" > "$D/backward_candidates.log" 2>&1
