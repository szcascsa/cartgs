#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TF_CAPTURE_PREFIX=${1:-phase_3}
trap 'rc=$?; printf "%s\n" "$rc" > "${TF_CAPTURE_PREFIX}_nsight.exit"' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64
TF_CHECKPOINT=/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply
nsys --version > "${TF_CAPTURE_PREFIX}_nsight_version.txt"
cuobjdump --dump-resource-usage build/libtf_cuda.a > "${TF_CAPTURE_PREFIX}_resource_usage.txt"
for TF_BACKEND in legacy cublas; do
  TF_TRACE="${TF_CAPTURE_PREFIX}_${TF_BACKEND}_nsys"
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --output="$TF_TRACE" \
    build/training_benchmark "$TF_CHECKPOINT" ../original_tf_cuda_audit_20261013/input.bin \
    "${TF_CAPTURE_PREFIX}_${TF_BACKEND}_profile" profile "$TF_BACKEND"
  nsys export --type=sqlite --output="${TF_TRACE}.sqlite" "${TF_TRACE}.nsys-rep"
  /home/prmi/anaconda3/envs/zwj/bin/python parse_nsight.py "$TF_TRACE"
done
