#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin:/usr/local/cuda-11.8/bin:$PATH
PRODUCTION=/home/prmi/zwj/mycartgs
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
PY=/home/prmi/anaconda3/envs/zwj/bin/python
test -f strict_complete.txt
for name in strict_build strict_checkpoint_1000 strict_real_219969 strict_random_1000; do
  test "$(cat "$name.exit")" = 0
done
"$PY" capture_environment.py round2_environment_before.json
if [ -f network_memory.csv ]; then mv network_memory.csv round1_profile_memory.csv; fi
echo 'backend,M,stage,allocated_before,allocated_peak,incremental_peak,reserved_peak' > network_memory.csv
./build-candidates/network_benchmark "$CHECKPOINT" "$FIXTURE" round2_network_benchmark.csv
mv network_memory.csv round2_network_memory.csv
for backend in legacy cublas warp cublas_strict; do
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "round2_mlp_$backend" \
    ./build-candidates/network_benchmark "$CHECKPOINT" "$FIXTURE" "round2_profile_$backend.csv" "$backend" profile > "round2_nsys_$backend.log" 2>&1
  nsys export --type sqlite --force-overwrite=true --output "round2_mlp_$backend.sqlite" "round2_mlp_$backend.nsys-rep" >> "round2_nsys_$backend.log" 2>&1
done
"$PY" capture_environment.py round2_environment_after.json
echo complete > round2_benchmark_complete.txt
