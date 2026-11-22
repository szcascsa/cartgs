#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'rc=$?; printf "%s\n" "$rc" > phase_5_nsight.exit' EXIT
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" --build build --target training_benchmark -j2
TF_CHECKPOINT=/home/prmi/zwj/mycartgs/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply
nsys --version > phase_5_nsight_version.txt
cuobjdump --dump-resource-usage build/libtf_cuda.a > phase_5_resource_usage.txt
for TF_MODE in regularized_adam regularized_preview; do
  TF_CAPTURE=profile
  if [[ "$TF_MODE" == regularized_preview ]]; then TF_CAPTURE=profile_preview; fi
  for TF_BACKEND in legacy cublas warp32; do
    TF_TRACE="phase_5_${TF_MODE}_${TF_BACKEND}_nsys"
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --output="$TF_TRACE" \
      build/training_benchmark "$TF_CHECKPOINT" ../original_tf_cuda_audit_20261013/input.bin \
      "phase_5_${TF_MODE}_${TF_BACKEND}_profile" cuda_optimizer "$TF_CAPTURE" "$TF_BACKEND"
    nsys export --type=sqlite --output="${TF_TRACE}.sqlite" "${TF_TRACE}.nsys-rep"
    /home/prmi/anaconda3/envs/zwj/bin/python parse_nsight.py "$TF_TRACE" "$TF_MODE"
  done
done
