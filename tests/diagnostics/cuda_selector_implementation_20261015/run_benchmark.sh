#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin:/usr/local/cuda-11.8/bin:$PATH
while [ ! -f verified_parity_complete.txt ]; do sleep 2; done
PRODUCTION=/home/prmi/zwj/mycartgs
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
PY=/home/prmi/anaconda3/envs/zwj/bin/python
"$PY" capture_environment.py environment_before_benchmark.json
echo 'Benchmark covers experimental MLP candidates only; failed parity forbids production promotion.'
echo 'backend,M,stage,allocated_before,allocated_peak,incremental_peak,reserved_peak' > network_memory.csv
./build-candidates/network_benchmark "$CHECKPOINT" "$FIXTURE" network_benchmark.csv
mv network_memory.csv network_memory_unprofiled.csv
for backend in legacy cublas warp; do
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "mlp_$backend" \
    ./build-candidates/network_benchmark "$CHECKPOINT" "$FIXTURE" "profile_$backend.csv" "$backend" profile > "nsys_$backend.log" 2>&1
  nsys export --type sqlite --force-overwrite=true --output "mlp_$backend.sqlite" "mlp_$backend.nsys-rep" >> "nsys_$backend.log" 2>&1
done
"$PY" capture_environment.py environment_after_benchmark.json
echo complete > benchmark_complete.txt
