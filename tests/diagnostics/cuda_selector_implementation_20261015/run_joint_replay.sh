#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_implementation_20261015
export LD_LIBRARY_PATH=$ROOT/lib:/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" --build "$D/build-caller" -j2 --target caller_replay > "$D/joint_replay_build.log" 2>&1
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
"$D/build-caller/caller_replay" "$D" joint 1000 > "$D/joint_fixed_map_replay.log" 2>&1
echo 0 > "$D/joint_fixed_map_replay.exit"
/home/prmi/anaconda3/envs/zwj/bin/python "$D/fixed_map_metrics.py" joint_fixed_map > "$D/joint_fixed_map_metrics.log" 2>&1
echo 0 > "$D/joint_fixed_map_metrics.exit"
cd "$D"
sha256sum joint_fixed_map_selected_ids.bin > joint_fixed_map_selected_ids.sha256
