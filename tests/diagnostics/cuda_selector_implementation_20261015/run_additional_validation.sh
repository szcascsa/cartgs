#!/usr/bin/env bash
set -u
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
PRODUCTION=/home/prmi/zwj/mycartgs
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
export LD_LIBRARY_PATH=$ROOT/lib:$PRODUCTION/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$CMAKE" -S "$D" -B "$D/build-caller" -DSELECTOR_INTEGRATION_VALIDATION=ON > "$D/additional_configure.log" 2>&1 &&
"$CMAKE" --build "$D/build-caller" -j2 --target caller_replay optimizer_candidate operator_trace > "$D/additional_build.log" 2>&1
build_exit=$?
echo "$build_exit" > "$D/additional_build.exit"
test "$build_exit" = 0 || exit "$build_exit"
# Stop only the queue coordinator, allowing its current train/evaluation child
# to finish naturally. The measured GPU work therefore never overlaps a run.
online_pid=$(pgrep -f '^/home/prmi/anaconda3/envs/zwj/bin/python (/home/prmi/zwj/selector-cuda-20261015/)?tests/diagnostics/cuda_selector_implementation_20261015/online_ab.py$' || true)
if [ -n "$online_pid" ]; then
  kill -STOP "$online_pid"
  trap 'kill -CONT "$online_pid" 2>/dev/null || true' EXIT
  while pgrep -P "$online_pid" >/dev/null; do
    # A completed subprocess is a zombie until its stopped parent reaps it.
    states=$(ps -o stat= --ppid "$online_pid" | tr -d 'Z \n')
    test -n "$states" || break
    sleep 2
  done
fi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$D/additional_gpu_before.txt"
"$D/build-caller/optimizer_candidate" "$D" > "$D/optimizer_candidate.log" 2>&1
echo $? > "$D/optimizer_candidate.exit"
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
for rows in 129 256 258 1000 1023 5000 10000 20000 50000 100000 109564 133003 200000; do
  "$D/build-candidates/selector_parity" --candidate cublas_strict --device cuda --rows "$rows" --steps 0 \
    --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out "$D/grid_strict_$rows.csv" > "$D/grid_strict_$rows.log" 2>&1
  echo $? > "$D/grid_strict_$rows.exit"
done
for backend in legacy cuda; do
  "$D/build-caller/operator_trace" "$CHECKPOINT" "$FIXTURE" "$backend" "$D/operator_trace_$backend.json" \
    > "$D/operator_trace_$backend.log" 2>&1
  echo $? > "$D/operator_trace_$backend.exit"
  "$NSYS" profile -f true --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output="$D/caller_$backend" "$D/build-caller/caller_replay" "$D" profile "$backend" \
    > "$D/caller_nsys_$backend.log" 2>&1
  echo $? > "$D/caller_nsys_$backend.exit"
  "$NSYS" export --type=sqlite --force-overwrite=true --output="$D/caller_$backend.sqlite" "$D/caller_$backend.nsys-rep" \
    >> "$D/caller_nsys_$backend.log" 2>&1
done
echo 0 > "$D/additional_pipeline.exit"
