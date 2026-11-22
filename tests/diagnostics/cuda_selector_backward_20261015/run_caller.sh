#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
P=/home/prmi/zwj/mycartgs
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
export SELECTOR_DIAGNOSTIC_BACKWARD=$1
DEST=$D/caller_$1${2:+_$2}
mkdir "$DEST"
/home/prmi/anaconda3/envs/zwj/bin/python - "$ROOT" "$DEST" <<'PY'
from pathlib import Path
import sys
root,dest=map(Path,sys.argv[1:])
for p in (root/'tests/diagnostics/cuda_selector_revised_20261015/replay_room').glob('current_*'):
    if p.name=='current_checkpoint' or p.suffix in ['.bin','.txt']:(dest/p.name).symlink_to(p)
PY
"$D/build/backward_caller" "$DEST" grid "$P/tests/diagnostics/original_selector_cuda_audit_20261013" \
 "$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt" > "$DEST/grid.log" 2>&1
"$D/build/backward_caller" "$DEST" benchmark > "$DEST/room.log" 2>&1
echo 0 > "$DEST/complete.exit"
