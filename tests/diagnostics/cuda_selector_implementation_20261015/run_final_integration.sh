#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
export LD_LIBRARY_PATH=$ROOT/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
cd "$ROOT"
# The first MLP-only queue is already stopped. Finish its existing child before
# replacing any experimental library; keep its output as preliminary evidence.
old_queue=$(pgrep -f '^/home/prmi/anaconda3/envs/zwj/bin/python tests/diagnostics/cuda_selector_implementation_20261015/online_ab.py$' || true)
if [ -n "$old_queue" ]; then
  while pgrep -P "$old_queue" >/dev/null; do
    states=$(ps -o stat= --ppid "$old_queue" | tr -d 'Z \n')
    test -n "$states" || break
    sleep 2
  done
  kill -TERM "$old_queue"
  kill -CONT "$old_queue" || true
fi
bash "$D/build_integration.sh" > "$D/final_integration_build.log" 2>&1
"$CMAKE" -S "$D" -B "$D/build-caller" -DSELECTOR_INTEGRATION_VALIDATION=ON > "$D/final_caller_configure.log" 2>&1
"$CMAKE" --build "$D/build-caller" -j2 --target caller_replay guard_transaction optimizer_candidate > "$D/final_caller_build.log" 2>&1
"$D/build-caller/guard_transaction" "$D/guard_transaction.csv" > "$D/guard_transaction.log" 2>&1
echo 0 > "$D/guard_transaction.exit"
"$D/build-caller/caller_replay" "$D" benchmark > "$D/final_caller_benchmark.log" 2>&1
"$D/build-caller/caller_replay" "$D" replay 1000 > "$D/final_fixed_map_replay.log" 2>&1
/home/prmi/anaconda3/envs/zwj/bin/python "$D/fixed_map_metrics.py" > "$D/final_fixed_map_metrics.log" 2>&1
bash "$D/run_additional_validation.sh" > "$D/final_additional_pipeline.log" 2>&1
echo 0 > "$D/final_integration_pipeline.exit"
/home/prmi/anaconda3/envs/zwj/bin/python "$D/online_ab.py" > "$D/final_online_pipeline.log" 2>&1
echo 0 > "$D/final_online_pipeline.exit"
