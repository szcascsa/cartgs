#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
export LD_LIBRARY_PATH=$ROOT/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake --build "$ROOT/build-integration" -j2 \
  --target joint_replay_selector_cuda_test > "$D/final_replay_integration_build.log" 2>&1
while [ ! -f "$D/caller_fullgrad_pipeline.exit" ]; do sleep 2; done
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
"$ROOT/bin/joint_replay_selector_cuda_test" cuda > "$D/final_replay_integration.log" 2>&1
echo 0 > "$D/final_replay_integration.exit"
