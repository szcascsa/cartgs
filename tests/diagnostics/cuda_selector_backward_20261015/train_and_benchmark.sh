#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
P=/home/prmi/zwj/mycartgs
PY=/home/prmi/anaconda3/envs/zwj/bin/python
export PATH=/home/prmi/anaconda3/envs/zwj/bin:/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 CXX=/usr/bin/g++-9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=$ROOT/python_extensions
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
strategy=$1
trap 'echo $? > "$D/candidate_${strategy}.exit"' EXIT
"$PY" -c "import json; s=json.load(open('$D/candidate_${strategy}_isolated/summary.json')); assert s.get('complete') and not s['failures']; assert all(v['all_exact'] for v in s['categories'].values())"
"$PY" "$D/gate.py" "$strategy" training > "$D/training_${strategy}.log" 2>&1
"$PY" -c "import json; s=json.load(open('$D/candidate_${strategy}_training/summary.json')); assert s.get('complete') and not s['failures']; assert all(v['all_exact'] for v in s['categories'].values())"
mkdir "$D/measure_${strategy}"
cd "$D/measure_${strategy}"
CHECKPOINT=$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$P/tests/diagnostics/original_selector_cuda_audit_20261013
for mode in -1 1 "$strategy"; do
 "$D/build/backward_benchmark" "$CHECKPOINT" "$FIXTURE" "strategy_${mode}.csv" "$mode" > "strategy_${mode}.log" 2>&1
done
