from pathlib import Path
import shutil
D=Path(__file__).resolve().parent
out=D/'phase2_safe'
assert not out.exists()
shutil.copytree(D/'phase2/source',out/'source')
(out/'raw').mkdir()
for f in ['benchmark.cpp','perf_scope.h']: shutil.copy2(D/'baseline'/f,out/f)
p=out/'source/src/change_distill.cpp'
s=p.read_text()
s=s.replace('        cd_transport_cuda::scatter(target, output, positions, product, correction);', '''        auto geometry_correction = correction;
        geometry_correction.changed_attributes &= ~toBits(GaussianAttrFlag::ROTATION);
        cd_transport_cuda::scatter(target, output, positions, product, geometry_correction);
        if (correction.changed_attributes & toBits(GaussianAttrFlag::ROTATION)) {
          const auto rotation = target.rotation.index_select(0, positions);
          const auto moved = quaternion_utils::normalize(quaternion_utils::multiply(
              correction.rotation_wxyz.reshape({1, 4}).expand({positions.numel(), 4}), rotation));
          output.rotation.index_copy_(0, positions, moved);
        }''')
p.write_text(s)
