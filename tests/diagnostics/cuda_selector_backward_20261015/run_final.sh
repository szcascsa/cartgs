#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
R=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
P=/home/prmi/zwj/mycartgs
C=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
CTEST=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest
PY=/home/prmi/anaconda3/envs/zwj/bin/python
export PATH=/home/prmi/anaconda3/envs/zwj/bin:/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/final.exit"' EXIT
"$PY" "$D/prepare_caller.py"
"$PY" "$D/prepare_profile.py"
"$PY" "$D/prepare_profile.py" --candidate
"$C" -S "$D" -B "$D/build" > "$D/final_configure.log" 2>&1
"$C" --build "$D/build" -j2 --target backward_benchmark backward_caller backward_profile_candidate > "$D/final_build.log" 2>&1
"$C" --build "$D/build" -j2 --target backward_boundary > "$D/boundary_final_build.log" 2>&1
"$D/build/backward_boundary" "$D/boundary_final.csv" "$1" 1 > "$D/boundary_final.log" 2>&1
"$C" --build "$ROOT/build-integration" -j2 --target change_distill_selector_cuda_test \
 drift_stability_selector_cuda_test joint_replay_selector_cuda_test gradient_constraint_test > "$D/native_build.log" 2>&1
"$C" --build "$R/build" -j2 --target guard_transaction > "$D/guard_build.log" 2>&1
"$CTEST" --test-dir "$ROOT/build-integration" --output-on-failure \
 -R '(change_distill_selector_cuda|drift_stability_selector_cuda|joint_replay_selector_cuda|^gradient_constraint$)' > "$D/native_tests.log" 2>&1
"$R/build/guard_transaction" "$D/final_guard.csv" > "$D/final_guard.log" 2>&1
export CUDA_HOME=/usr/local/cuda-11.8 CXX=/usr/bin/g++-9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=$ROOT/python_extensions SELECTOR_DIAGNOSTIC_TAG=final
"$PY" "$D/gate.py" "$1" all > "$D/final_direct.log" 2>&1
"$PY" -c "import json; s=json.load(open('$D/candidate_${1}_all_final/summary.json')); assert s.get('complete') and not s['failures']; assert all(v['all_exact'] for v in s['categories'].values())"
CHECKPOINT=$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$P/tests/diagnostics/original_selector_cuda_audit_20261013
mkdir "$D/final_measure"
cd "$D/final_measure"
sha256sum "$D/build/backward_benchmark" "$D/build/backward_caller" "$ROOT/lib/libgaussian_mapper.so" > binaries.sha256
nvidia-smi -q > gpu_before.txt
for mode in -1 1 6 "$1"; do
 [ -f "strategy_${mode}.csv" ] && continue
 "$D/build/backward_benchmark" "$CHECKPOINT" "$FIXTURE" "strategy_${mode}.csv" "$mode" > "strategy_${mode}.log" 2>&1
done
for pass in 2 3; do
 mkdir "repeat_$pass"
 cd "repeat_$pass"
 for mode in "$1" 1; do
  "$D/build/backward_benchmark" "$CHECKPOINT" "$FIXTURE" "strategy_${mode}.csv" "$mode" 219969 > "strategy_${mode}.log" 2>&1
 done
 cd ..
done
bash "$D/run_caller.sh" 1 final
bash "$D/run_caller.sh" "$1" final
bash "$D/run_profiles.sh" "$1"
nvidia-smi -q > gpu_after.txt
