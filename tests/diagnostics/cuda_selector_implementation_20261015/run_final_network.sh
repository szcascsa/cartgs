#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PATH=/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin:$PATH
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
PRODUCTION=/home/prmi/zwj/mycartgs
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
"$CMAKE" --build build-candidates -j2 > final_network_build.log 2>&1
./build-candidates/selector_parity --candidate cublas_strict --device cuda --rows 257 --steps 1000 \
  --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out final_strict_1000.csv > final_strict_1000.log 2>&1
./build-candidates/selector_parity --candidate cublas_strict --device cuda --rows 219969 --steps 0 \
  --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --out final_strict_219969.csv > final_strict_219969.log 2>&1
echo 0 > final_network_parity.exit
# Serialize with the renderer/caller experiment before launching GPU measurements.
while [ ! -f caller_benchmark.exit ]; do sleep 2; done
while pgrep -f '[/]caller_replay .* replay ' > /dev/null; do sleep 2; done
bash run_strict_benchmark.sh > final_network_benchmark.log 2>&1
echo 0 > final_network_benchmark.exit
