"""只清理已通过候选产生的未使用代码，保留独立实验快照。"""
from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'final'
assert not out.exists()
shutil.copytree(D/'phase3_strict/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill_transport.cu'
s=p.read_text()
prefix=s[:s.index('__global__ void gatherXYZ')]
suffix=s[s.index('__global__ void compactXYZKernel'):]
s=prefix+'''// Float metadata rows: matrix[16], wxyz[4], scale, log(scale), flags.
// The reduction trees and explicit cross-product FMAs match LibTorch 2.3.1.
// Keep --fmad=false: only the original cross product uses explicit FMA.
'''+suffix
p.write_text(s)
p=out/'source/include/change_distill_transport_cuda.h'
p.write_text('''#pragma once
#include "include/gaussian_transform_ops.h"

namespace cd_transport_cuda {
// positions and groups are aligned contiguous int64 vectors in stable group order.
torch::Tensor compactXYZ(const torch::Tensor& xyz, const torch::Tensor& positions,
                         const torch::Tensor& groups, const torch::Tensor& metadata);
void scatterCompact(const GaussianAttrState& input, const GaussianAttrState& output,
                    const torch::Tensor& positions, const torch::Tensor& groups,
                    const torch::Tensor& metadata, const torch::Tensor& product);
}
''')
p=out/'source/src/change_distill.cpp'
s=p.read_text().replace('''        const auto& correction=groups_[segment.group];
        if''','''        const auto& correction=groups_[segment.group];
        TORCH_CHECK(std::isfinite(correction.scale) && correction.scale > 0.0f,
                    "Gaussian correction requires a positive finite scale");
        if''')
s=s.replace('''            missing.numel() == batch.numel() ? plan : nullptr);''','''            // Boolean missing filtering preserves order. Equal lengths prove
            // the callback sees the complete planned batch; partial misses use
            // the generic plan builder instead.
            missing.numel() == batch.numel() ? plan : nullptr);''')
p.write_text(s)
p=out/'source/CMakeLists.txt'
s=p.read_text().replace('    src/transform_field_cuda.cpp src/transform_field_cuda.cu',
                        '    src/change_distill_transport.cu\n    src/transform_field_cuda.cpp src/transform_field_cuda.cu',1)
p.write_text(s)
