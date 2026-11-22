#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
P=/home/prmi/zwj/mycartgs
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
CHECKPOINT=$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$P/tests/diagnostics/original_selector_cuda_audit_20261013
for strategy in 1 "$1"; do
 DEST=$D/profile_final_$strategy
 mkdir "$DEST"
 for m in 10000 50000 100000 219969; do
  "$NSYS" profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
   --capture-range=cudaProfilerApi --capture-range-end=stop --output="$DEST/profile_$m" \
   "$D/build/backward_profile_candidate" "$CHECKPOINT" "$FIXTURE" "$m" scalar "$strategy" > "$DEST/log_$m.txt" 2>&1
  "$NSYS" export --type=sqlite --output="$DEST/profile_$m.sqlite" "$DEST/profile_$m.nsys-rep" > "$DEST/export_$m.txt" 2>&1
 done
 /home/prmi/anaconda3/envs/zwj/bin/python "$D/parse_profile.py" "$DEST" > "$DEST/breakdown.txt"
done
DEST=$D/profile_final_row
mkdir "$DEST"
"$NSYS" profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
 --capture-range=cudaProfilerApi --capture-range-end=stop --output="$DEST/profile_219969" \
 "$D/build/backward_profile_candidate" "$CHECKPOINT" "$FIXTURE" 219969 row "$1" > "$DEST/log.txt" 2>&1
"$NSYS" export --type=sqlite --output="$DEST/profile_219969.sqlite" "$DEST/profile_219969.nsys-rep" > "$DEST/export.txt" 2>&1
/home/prmi/anaconda3/envs/zwj/bin/python "$D/parse_profile.py" "$DEST" > "$DEST/breakdown.txt"
