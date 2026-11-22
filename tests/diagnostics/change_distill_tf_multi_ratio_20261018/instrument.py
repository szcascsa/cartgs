"""只修改隔离副本；native 源码不含计时/校验 hooks。"""
from pathlib import Path
import shutil,sys
D=Path(__file__).resolve().parent;stage=D/sys.argv[1];kind=sys.argv[2]
OLD=Path('/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017')
dst=stage/('instrumented' if kind=='profile' else 'verify');shutil.copytree(stage/'source',dst,dirs_exist_ok=True)
def edit(name,fn):
    p=dst/name;p.write_text(fn(p.read_text()))
def op(s,expr,label):
    return s.replace(expr,'CD_OP("'+label+'", '+expr+')')
if kind=='profile':
    def cd(s):
        s='#include "perf_scope.h"\n'+s
        s=s.replace('  if (selector_indices.numel() > 0 &&','  cdperf::branch="selector";\n  if (selector_indices.numel() > 0 &&',1)
        s=s.replace('  if (tf_indices.numel() > 0 &&','  cdperf::branch="tf";\n  if (tf_indices.numel() > 0 &&',1)
        s=s.replace('  auto qs = quaternion_utils::normalize(student.rotation);','  cdperf::Scope loss_scope("loss_construction");\n  auto qs = quaternion_utils::normalize(student.rotation);')
        s=s.replace('  return transport.apply(std::move(target), indices, plan);','  cdperf::Scope scope("geometry_transport");\n  return transport.apply(std::move(target), indices, plan);')
        expressions=[('TargetTransport(event)','transport_metadata'),('transport.prepare(indices, batch_size)','transport_plan'),
          ('capture.teacher_tf.ptr()->applyLogResidual(\n            capture.old_state.select(missing), ratio)','teacher_forward'),
          ('context.tf->applyLogResidual(current.select(batch), ratio)','student_forward'),
          ('context.tf->applyLogResidualMany(current.select(batch), context.ratios)','student_many_forward'),
          ('current.select(batch)','student_input_gather'),('evaluate(tf_indices)','pre_evaluation'),('evaluate(tf_indices, true)','post_evaluation'),
          ('evaluate(audit)','audit_pre'),('evaluate(audit, true)','audit_post'),
          ('loss[j].item<double>()','evaluation_scalar_readback'),('total.backward()','backward'),('optimizer->zero_grad()','zero_grad'),
          ('torch::stack(scalars).to(torch::kCPU).to(torch::kFloat64)','evaluation_scalar_readback'),
          ('tf_cuda::tfCudaAdamDirect(*optimizer, {0, 1})','adam_step'),('torch::isfinite(total).item<bool>()','finite_readback'),('total.item<double>()','positive_readback'),
          ('torch::cat({target.xyz, target.log_scaling, target.rotation}, 1)','teacher_pack')]
        for expr,label in expressions:s=op(s,expr,label)
        return s
    edit('src/change_distill.cpp',cd)
    def tf(s):
        s='#include "perf_scope.h"\n'+s
        exprs=[('tf_cuda::query(xyz, ratio_tensor, aabbMin(), aabbMax(), gridParameters())','hexplane_forward'),
         ('tf_cuda::queryMany(xyz, ratios, aabbMin(), aabbMax(), gridParameters())','hexplane_many_forward'),
         ('tf_cuda::neural(feature, mlpParameters(), mlp_backend)','mlp_forward'),
         ('tf_cuda::neural(torch::cat(features, 0), mlpParameters(), mlp_backend)','mlp_many_forward'),
         ('tf_cuda::neuralMany(features, mlpParameters(), mlp_backend)','mlp_many_forward'),
         ('torch::nonzero(insideMask(state.xyz)).reshape({-1})','student_domain_filter'),('state.select(indices)','student_attribute_gather'),
         ('tf_cuda::output(selected.xyz, selected.log_scaling, selected.rotation,\n        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw}, true)','output_forward')]
        for expr,label in exprs:s=op(s,expr,label)
        return s
    edit('src/transform_field.cpp',tf)
    def autograd(s):
        s='#include "perf_scope.h"\n'+s
        for token,label in [('    return queryBackward(ctx->get_saved_variables(), gradients.at(0).contiguous());','hexplane_backward'),
                            ('    auto saved=ctx->get_saved_variables();','mlp_backward'),('    auto result=outputBackward(','output_backward')]:
            s=s.replace(token,'    cdperf::Scope scope("'+label+'");\n'+token)
        return s
    edit('src/transform_field_cuda.cpp',autograd)
    def cache(s):
        s='#include "perf_scope.h"\n'+s
        for expr,label in [('indices.index({~ready_.index_select(0, indices)})','cache_missing_indices'),('evaluate(missing).detach()','cache_miss'),('values_.index_select(0, indices)','cache_gather')]:s=op(s,expr,label)
        return s
    edit('include/frozen_teacher_target_cache.h',cache)
else:
    shutil.copy2(OLD/'verify_scope.h',stage/'verify_scope.h')
    header=(stage/'verify_scope.h').read_text()+'''
namespace cdverify {
inline int64_t features=0;
inline std::vector<int64_t> backward_order;
inline void feature(const torch::Tensor& x) {
  if(!torch::GradMode::is_enabled() || !x.requires_grad()) return;
  const auto i=features++;
  if(i>=4) return;
  archive.write("feature_"+std::to_string(i),x.detach().clone());
  x.register_hook([i](torch::Tensor g){
    archive.write("feature_gradient_"+std::to_string(i),g.detach().clone());
    backward_order.push_back(i);return g;
  });
}
inline void metrics(const std::string& name,const std::array<double,3>& x) {
  archive.write(name+"_"+std::to_string(tf_steps),torch::tensor(std::vector<double>(x.begin(),x.end()),torch::kFloat64));
}
}
'''
    (stage/'verify_scope.h').write_text(header)
    def cd(s):
        s='#include "verify_scope.h"\n'+s
        s=s.replace('return torch::cat({target.xyz, target.log_scaling, target.rotation}, 1);','auto packed = torch::cat({target.xyz, target.log_scaling, target.rotation}, 1); cdverify::target(packed); return packed;')
        for branch in ['selector','tf']:
            token='          total.backward();' if branch=='selector' else '            total.backward();'
            s=s.replace(token,token+'\n cdverify::step(total,context.'+branch+'->parameters(),'+str(branch=='tf').lower()+');',1)
            token='++report.'+branch+'_steps;';s=s.replace(token,token+'\n cdverify::updated(context.'+branch+'->parameters(),'+str(branch=='tf').lower()+');')
        for token,name in [('const auto pre = evaluate(tf_indices);','pre'),('post = evaluate(tf_indices, true);','post'),('const auto audit_pre = evaluate(audit);','audit_pre'),('const auto audit_post = evaluate(audit, true);','audit_post')]:
            var={'pre':'pre','post':'post','audit_pre':'audit_pre','audit_post':'audit_post'}[name]
            s=s.replace(token,token+'\n cdverify::metrics("'+name+'",'+var+');')
        return s
    edit('src/change_distill.cpp',cd)
    def tf(s):
        s='#include "verify_scope.h"\n'+s
        s=s.replace('    // Same-machine complete','    cdverify::feature(feature);\n    // Same-machine complete')
        s=s.replace('      const auto& feature = features[r];','      const auto& feature = features[r];\n      cdverify::feature(feature);')
        return s
    edit('src/transform_field.cpp',tf)
    bench=(stage/'benchmark.cpp').read_text().replace('int main(int argc,char** argv)','int benchmarkMain(int argc,char** argv)')
    bench+='''
int main(int argc,char** argv) {
  int result=benchmarkMain(argc,argv);
  if(!result) {
    cdverify::archive.write("backward_order",torch::tensor(cdverify::backward_order,torch::kLong));
    cdverify::archive.save_to((fs::path(argv[2])/"verification.pt").string());
  }
  return result;
}
'''
    (stage/'benchmark_verify.cpp').write_text(bench)
print('INSTRUMENT',stage.name,kind)
