#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
P=/home/prmi/zwj/mycartgs
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/paired.exit"' EXIT
CHECKPOINT=$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$P/tests/diagnostics/original_selector_cuda_audit_20261013
"$D/build/backward_paired" "$CHECKPOINT" "$FIXTURE" "$D/paired.csv" > "$D/paired.log" 2>&1
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
"$NSYS" profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
 --capture-range=cudaProfilerApi --capture-range-end=stop --output="$D/paired_fb" \
 "$D/build/backward_paired" "$CHECKPOINT" "$FIXTURE" "$D/paired_profile.csv" profile > "$D/paired_nsys.log" 2>&1
"$NSYS" export --type=sqlite --output="$D/paired_fb.sqlite" "$D/paired_fb.nsys-rep" > "$D/paired_export.log" 2>&1
/home/prmi/anaconda3/envs/zwj/bin/python "$D/parse_paired.py" > "$D/paired_profile_summary.log"
