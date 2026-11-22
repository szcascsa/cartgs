#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
PRODUCTION=/home/prmi/zwj/mycartgs
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
"$CMAKE" --build "$D/build-caller" -j2 --target caller_replay > "$D/caller_fullgrad_build.log" 2>&1
online_pid=$(pgrep -f '^/home/prmi/anaconda3/envs/zwj/bin/python (/home/prmi/zwj/selector-cuda-20261015/)?tests/diagnostics/cuda_selector_implementation_20261015/online_ab.py$' || true)
if [ -n "$online_pid" ]; then
  kill -STOP "$online_pid"
  trap 'kill -CONT "$online_pid" 2>/dev/null || true' EXIT
  while pgrep -P "$online_pid" >/dev/null; do
    states=$(ps -o stat= --ppid "$online_pid" | tr -d 'Z \n')
    test -n "$states" || break
    sleep 2
  done
fi
"$D/build-caller/caller_replay" "$D" benchmark > "$D/caller_fullgrad_benchmark.log" 2>&1
"$D/build-caller/caller_replay" "$D" grid "$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013" \
  "$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt" > "$D/caller_fullgrad_grid.log" 2>&1
for backend in legacy cuda; do
  "$NSYS" profile -f true --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output="$D/caller_$backend" "$D/build-caller/caller_replay" "$D" profile "$backend" \
    > "$D/caller_nsys_$backend.log" 2>&1
  "$NSYS" export --type=sqlite --force-overwrite=true --output="$D/caller_$backend.sqlite" "$D/caller_$backend.nsys-rep" \
    >> "$D/caller_nsys_$backend.log" 2>&1
done
python3 "$D/parse_caller_nsight.py" > "$D/caller_fullgrad_nsight.log"
echo 0 > "$D/caller_fullgrad_pipeline.exit"
