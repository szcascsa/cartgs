"""只给冻结副本添加 scopes 和 correctness hooks。"""
from pathlib import Path
import shutil
import sys

D = Path(__file__).resolve().parent
OLD = D.parent/'change_distill_optimization_20261017'
if not OLD.exists(): OLD = Path('/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017')
stage = D/sys.argv[1]
(stage/'raw').mkdir(exist_ok=True)
kind = sys.argv[2]
dst = stage/('instrumented' if kind=='profile' else 'verify')
shutil.copytree(stage/'source', dst, dirs_exist_ok=True)
p = dst/'src/change_distill.cpp'
s = p.read_text()
if kind == 'profile':
    # Reuse the audited expression wrappers and extend only transport scopes.
    code = (OLD/'stage_c/generate_profile_stage.py').read_text()
    code = code.replace("D = Path(__file__).resolve().parent", 'D = Path('+repr(str(stage))+')')
    if sys.argv[1] != 'baseline':
        code = code.replace('return transport.apply(std::move(target), indices);',
                            'return transport.apply(std::move(target), indices, plan);')
    exec(compile(code, str(OLD/'stage_c/generate_profile_stage.py'), 'exec'))
    s = p.read_text()
    for expr, label in [
        ('indices.cpu().contiguous()', 'membership_ids_d2h'),
        ('torch::tensor(packed, torch::kLong).to(indices.device())', 'membership_positions_h2d'),
        ('applyGaussianTransform(target.select(positions_g), groups_[g])', 'group_transform'),
        ('target.xyz.index_copy(0, positions_g, moved.xyz)', 'xyz_index_copy'),
        ('target.log_scaling.index_copy(0, positions_g, moved.log_scaling)', 'scale_index_copy'),
        ('target.rotation.index_copy(0, positions_g, moved.rotation)', 'rotation_index_copy'),
        ('transport.prepare(indices, batch_size)', 'transport_plan_population'),
        ('cd_transport_cuda::gatherScaledXYZ(target.xyz, positions, correction.scale)', 'transport_compact_gather'),
        ('torch::matmul(scaled, correction.transform_4x4.slice(0, 0, 3).slice(1, 0, 3))', 'transport_xyz_gemm'),
        ('cd_transport_cuda::scatter(target, output, positions, product, correction)', 'transport_fused_postprocess_scatter'),
        ('cd_transport_cuda::compactXYZ(target.xyz,positions,groups,metadata_)', 'transport_compact_gather'),
        ('cd_transport_cuda::scatterCompact(target,output,positions,groups,metadata_,product)', 'transport_fused_postprocess_scatter'),
    ]:
        if expr in s: s = s.replace(expr, 'CD_OP("'+label+'", '+expr+')')
    s=s.replace('          torch::mm_out(result,', '          cdperf::Scope gemm_scope("transport_xyz_gemm");\n          torch::mm_out(result,')
    s = s.replace('    const auto* ids = host_indices.data_ptr<int64_t>();',
                  '    cdperf::Scope membership_scope("membership_cpu_and_transport");\n    const auto* ids = host_indices.data_ptr<int64_t>();')
    p.write_text(s)
else:
    shutil.copy2(OLD/'verify_scope.h', stage/'verify_scope.h')
    s = '#include "verify_scope.h"\n'+s
    s = s.replace('        auto target = transportTarget(', '        cdverify::archive.write("ids_"+std::to_string(cdverify::targets), missing.detach().clone());\n        auto target = transportTarget(')
    token = 'return torch::cat({target.xyz, target.log_scaling, target.rotation}, 1);'
    assert token in s
    s = s.replace(token, 'auto packed_target = torch::cat({target.xyz, target.log_scaling, target.rotation}, 1); cdverify::target(packed_target); return packed_target;')
    for branch in ['selector', 'tf']:
        token = '          total.backward();' if branch=='selector' else '            total.backward();'
        s = s.replace(token, token+'\n cdverify::step(total, context.'+branch+'->parameters(), '+str(branch=='tf').lower()+');', 1)
        token = '++report.'+branch+'_steps;'
        s = s.replace(token, token+'\n cdverify::updated(context.'+branch+'->parameters(), '+str(branch=='tf').lower()+');')
    p.write_text(s)
    bench = (stage/'benchmark.cpp').read_text().replace('int main(int argc,char** argv)', 'int benchmarkMain(int argc,char** argv)')
    reference = (D/'baseline/source/src/change_distill.cpp').read_text()
    reference = reference[reference.index('class TargetTransport'):reference.index('std::array<torch::Tensor, 3> tfErrors')]
    reference = reference.replace('TargetTransport','ReferenceTransport').replace('transportTarget','referenceTransportImpl')
    bench += '\n'+reference+'\nGaussianAttrState referenceTransport(GaussianAttrState x, const torch::Tensor& ids, const ChangeEvent& e) { return ReferenceTransport(e).apply(x,ids); }\n'
    bench += (D/'transport_verify.cpp').read_text()
    bench += '''
int main(int argc, char** argv) {
  torch::set_num_threads(4); Snapshot snapshot(argv[1]); fs::create_directories(argv[2]);
  writeJson(fs::path(argv[2])/"transport_parity.json",transportParity(snapshot));
  const int result=benchmarkMain(argc,argv);
  if(!result) cdverify::archive.save_to((fs::path(argv[2])/"verification.pt").string());
  return result;
}
'''
    (stage/'benchmark_verify.cpp').write_text(bench)
