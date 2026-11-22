#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-11.8/bin:$PATH
PRODUCTION=/home/prmi/zwj/mycartgs
CHECKPOINT=$PRODUCTION/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$PRODUCTION/tests/diagnostics/original_selector_cuda_audit_20261013
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
sha256sum selector_parity.cpp network_benchmark.cpp ../../../src/selector_cuda.cpp \
  ../../../src/selector_cuda.cu ../../../include/selector_cuda.h > verified_source_hashes.txt
"$CMAKE" --build build-candidates --clean-first -j2 > verified_build.log 2>&1
run_case() {
  local name=$1 backend=$2 rows=$3 steps=$4
  set +e
  ./build-candidates/selector_parity --device cuda --candidate "$backend" \
    --checkpoint "$CHECKPOINT" --fixture "$FIXTURE" --rows "$rows" \
    --steps "$steps" --out "$name.csv" > "$name.log" 2>&1
  local status=$?
  set -e
  echo "$status" > "$name.exit"
  echo "$name exit=$status"
}
run_case phase1_verified_legacy_1000 legacy 257 1000
run_case phase1_verified_real_219969 legacy 219969 0
run_case phase2_verified_cublas_1000 cublas 257 1000
run_case phase2_verified_cublas_real_219969 cublas 219969 0
run_case phase3_verified_warp_1000 warp 257 1000
run_case phase3_verified_warp_real_219969 warp 219969 0
echo complete > verified_parity_complete.txt
