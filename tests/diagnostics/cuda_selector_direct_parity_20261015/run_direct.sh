#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_direct_parity_20261015
REVISED=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
PRODUCTION=/home/prmi/zwj/mycartgs
PYTHON=/home/prmi/anaconda3/envs/zwj/bin/python
CTEST=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest
export PATH=/home/prmi/anaconda3/envs/zwj/bin:/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 CXX=/usr/bin/g++-9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=$ROOT/python_extensions
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/pipeline.exit"' EXIT
cd "$D"
"$PYTHON" direct.py > direct.log 2>&1
echo 0 > direct.exit
"$REVISED/build/guard_transaction" "$D/guard_transaction.csv" > guard_transaction.log 2>&1
"$CTEST" --test-dir "$ROOT/build-integration" --output-on-failure \
 -R '(change_distill_selector_cuda|drift_stability_selector_cuda|joint_replay_selector_cuda|^gradient_constraint$)' > native_special_paths.log 2>&1
if "$REVISED/build/selector_parity" --candidate cublas_strict --device cuda --rows 257 --steps 1000 \
 --checkpoint "$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt" \
 --fixture "$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013" --out "$D/native_1000.csv" > native_1000.log 2>&1; then
 echo 0 > native_diagnostic.exit
else
 echo $? > native_diagnostic.exit
fi
# This historical harness has adversarial/exact gates; it is diagnostic only.
# The seven-checkpoint direct.py is the actual numerical/training gate.
echo 0 > correctness.exit
# Performance starts only after the direct correctness phase succeeds.
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
for backend in legacy cublas_strict; do
 "$REVISED/build/network_benchmark" "$CHECKPOINT" "$FIXTURE" "$D/network_$backend.csv" "$backend" > "$D/network_$backend.log" 2>&1
done
mkdir "$D/caller"
"$PYTHON" - <<'PY'
from pathlib import Path
root=Path('/home/prmi/zwj/selector-cuda-20261015/tests/diagnostics')
for p in (root/'cuda_selector_revised_20261015/replay_room').glob('current_*'):
 if p.name=='current_checkpoint' or p.suffix in ['.bin','.txt']:
  (root/'cuda_selector_direct_parity_20261015/caller'/p.name).symlink_to(p)
PY
"$REVISED/build/caller_replay" "$D/caller" grid "$FIXTURE" "$CHECKPOINT" > caller_grid.log 2>&1
"$REVISED/build/caller_replay" "$D/caller" benchmark > caller_room.log 2>&1
echo 0 > performance.exit
