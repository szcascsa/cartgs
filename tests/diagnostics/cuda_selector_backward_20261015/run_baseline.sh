#!/usr/bin/env bash
set -eu
ROOT=/home/prmi/zwj/selector-cuda-20261015
D=$ROOT/tests/diagnostics/cuda_selector_backward_20261015
R=$ROOT/tests/diagnostics/cuda_selector_revised_20261015
P=/home/prmi/zwj/mycartgs
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
export PATH=/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$ROOT/lib:$P/third_party/libtorch-cu118/libtorch/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}
trap 'echo $? > "$D/baseline.exit"' EXIT
CHECKPOINT=$P/results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply/selector.pt
FIXTURE=$P/tests/diagnostics/original_selector_cuda_audit_20261013
mkdir "$D/baseline"
cd "$D/baseline"
sha256sum "$R/build/network_benchmark" > binary.sha256
nvidia-smi -q > gpu.txt
for backend in legacy cublas_strict; do
 "$R/build/network_benchmark" "$CHECKPOINT" "$FIXTURE" "$backend.csv" "$backend" > "$backend.log" 2>&1
done
"$CMAKE" -S "$D" -B "$D/build" -GNinja -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 \
 -DTorch_DIR="$P/third_party/libtorch-cu118/libtorch/share/cmake/Torch" > "$D/configure.log" 2>&1
"$CMAKE" --build "$D/build" -j2 > "$D/build.log" 2>&1
NSYS=/usr/local/cuda-11.8/nsight-systems-2022.4.2/bin/nsys
for m in 10000 50000 100000 219969; do
 "$NSYS" profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --output="$D/baseline/profile_$m" \
  "$D/build/backward_profile" "$CHECKPOINT" "$FIXTURE" "$m" scalar > "nsys_$m.log" 2>&1
 "$NSYS" export --type=sqlite --output="$D/baseline/profile_$m.sqlite" "$D/baseline/profile_$m.nsys-rep" > "export_$m.log" 2>&1
done
/usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu --target-processes all --profile-from-start off \
 --launch-count 15 --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread \
 --csv --log-file "$D/baseline/ncu.csv" "$D/build/backward_profile" "$CHECKPOINT" "$FIXTURE" 219969 scalar \
 > "$D/baseline/ncu.log" 2>&1 || echo $? > "$D/baseline/ncu.exit"
echo 0 > "$D/profile_complete.exit"
