"""只在冻结副本添加计时/shape hooks；保留所有训练表达式与控制流。"""
from pathlib import Path
import difflib
import json
import shutil

D = Path(__file__).resolve().parent
S = D / 'source'
G = D / 'instrumented'
shutil.copytree(S, G, dirs_exist_ok=True)
changes = []


def write(relative, text):
    before = (S / relative).read_text(encoding='utf-8')
    (G / relative).write_text(text, encoding='utf-8', newline='\n')
    changes.extend(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                      fromfile=relative, tofile='instrumented/' + relative))


def wrap(text, expression, name, expected=None):
    count = text.count(expression)
    assert count > 0 and (expected is None or count == expected), (name, count, expected)
    return text.replace(expression, 'CD_OP("' + name + '", ' + expression + ')')


s = (S / 'src/change_distill.cpp').read_text(encoding='utf-8')
s = '#include "perf_scope.h"\n' + s
for signature, label in [
    ('  TORCH_CHECK(!learning_rates.empty() &&', 'adam_construction'),
    ('  if (scope == DistillScope::ALL_SNAPSHOT_MATURE)', 'scope_indices'),
    ('  torch::NoGradGuard no_grad;\n  auto ids = event.correction_group_id', 'geometry_transport'),
    ('  auto qs = quaternion_utils::normalize(student.rotation);', 'loss_construction'),
    ('  auto after = current.select(indices);', 'selector_loss_call'),
    ('  std::vector<int64_t> order(indices.numel());', 'shuffle_cpu_h2d_gather'),
    ('  if (!config_.enable_unaffected_anchors || corrected_count == 0)', 'anchor_sampling'),
]:
    assert s.count(signature) == 1, signature
    s = s.replace(signature, '  cdperf::Scope timer_' + label + '("' + label + '");\n' + signature)
s = s.replace('  if (selector_indices.numel() > 0 &&',
              '  cdperf::branch = "selector";\n  if (selector_indices.numel() > 0 &&', 1)
s = s.replace('  if (tf_indices.numel() > 0 &&',
              '  cdperf::branch = "tf";\n  if (tf_indices.numel() > 0 &&', 1)
s = s.replace('    for (int64_t epoch = 0; epoch < config_.selectorEpochs(); ++epoch) {',
              '    for (int64_t epoch = 0; epoch < config_.selectorEpochs(); ++epoch) {\n'
              '      cdperf::Scope epoch_scope("epoch_" + std::to_string(epoch));')
s = s.replace('      for (int64_t epoch = 0; epoch < config_.tfEpochs(); ++epoch) {',
              '      for (int64_t epoch = 0; epoch < config_.tfEpochs(); ++epoch) {\n'
              '        cdperf::Scope epoch_scope("epoch_" + std::to_string(epoch));')
s = s.replace('        optimizer->zero_grad();',
              '        cdperf::Scope batch_scope("training_batch");\n        optimizer->zero_grad();', 1)
s = s.replace('          optimizer->zero_grad();',
              '          cdperf::Scope batch_scope("training_batch");\n          optimizer->zero_grad();', 1)
expressions = [
    ('current.select(indices)', 'gaussian_gather'),
    ('capture.old_state.select(missing)', 'teacher_gaussian_gather'),
    ('torch::full({indices.numel(), 1}, ratio, after.xyz.options())', 'ratio_tensor'),
    ('torch::full({missing.numel(), 1}, ratio, before.xyz.options())', 'teacher_ratio_tensor'),
    ('capture.teacher_selector.ptr()->forwardProbability(\n        before.xyz, before.rotation, before.log_scaling.exp(), teacher_ratios, temperature)', 'teacher_forward'),
    ('student->forwardProbability(\n      after.xyz, after.rotation, after.log_scaling.exp(), ratios, temperature)', 'student_forward'),
    ('(probability - teacher).abs().mean()', 'loss_construction'),
    ('evaluate(selector_indices)', 'primary_evaluation'),
    ('evaluate(audit_anchors)', 'anchor_evaluation'),
    ('evaluate(tf_indices)', 'primary_evaluation'),
    ('evaluate(audit)', 'untouched_evaluation'),
    ('stratumBatch(corrected, b, batches)', 'primary_batch_view'),
    ('stratumBatch(anchors, b, batches)', 'anchor_batch_view'),
    ('current.xyz.index_select(0, tf_indices)', 'filter_gaussian_gather'),
    ('context.tf->insideMask(CD_OP("filter_gaussian_gather", current.xyz.index_select(0, tf_indices)))', 'outside_identity_filter'),
    ('tf_indices.index({inside})', 'filtered_indices'),
    ('capture.old_aabb.diagonal().item<double>()', 'aabb_diagonal_d2h'),
    ('capture.teacher_tf.ptr()->applyLogResidual(\n            CD_OP("teacher_gaussian_gather", capture.old_state.select(missing)), ratio)', 'teacher_forward'),
    ('torch::cat({target.xyz, target.log_scaling, target.rotation}, 1)', 'target_pack'),
    ('current.select(batch)', 'gaussian_gather'),
    ('context.tf->applyLogResidual(CD_OP("gaussian_gather", current.select(batch)), ratio)', 'student_forward'),
    ('optimizer->zero_grad()', 'zero_grad'),
    ('total.backward()', 'backward'),
    ('tf_cuda::tfCudaAdamDirect(*optimizer, {0})', 'adam_step'),
    ('tf_cuda::tfCudaAdamDirect(*optimizer, {0, 1})', 'adam_step'),
    ('optimizer->step()', 'adam_step'),
    ('torch::isfinite(total).item<bool>()', 'finite_check_d2h'),
    ('total.item<double>()', 'positive_check_d2h'),
    ('torch::tensor(order, torch::TensorOptions().dtype(torch::kLong))\n                         .to(indices.device())', 'permutation_h2d'),
    ('indices.index_select(0, permutation)', 'shuffle_index_gather'),
]
for expression, name in expressions:
    s = wrap(s, expression, name)
# Preserve lazy creation/initialization and evaluate them separately from optimizer step.
s = s.replace('auto optimizer = independentAdam(', 'auto optimizer = independentAdam(')
s = s.replace('    CD_OP("zero_grad", optimizer->zero_grad());\n    std::ostringstream summary;',
              '    CD_OP("zero_grad", optimizer->zero_grad());\n    cdperf::optimizer(*optimizer);\n    std::ostringstream summary;')
s = s.replace('      CD_OP("zero_grad", optimizer->zero_grad());\n    } else {',
              '      CD_OP("zero_grad", optimizer->zero_grad());\n      cdperf::optimizer(*optimizer);\n    } else {')
# Shape probes are metadata reads, and only the instrumented binary executes them.
s = s.replace('  auto probability = CD_OP', '  cdperf::tensor("selector/current_xyz", after.xyz);\n'
              '  cdperf::tensor("selector/teacher_probability", teacher);\n  auto probability = CD_OP')
s = s.replace('  return CD_OP("loss_construction", (probability',
              '  cdperf::tensor("selector/student_probability", probability);\n'
              '  return CD_OP("loss_construction", (probability')
s = s.replace('      return tfErrors(',
              '      cdperf::tensor("tf/target_xyz",target.xyz);\n'
              '      cdperf::tensor("tf/target_log_scale",target.log_scaling);\n'
              '      cdperf::tensor("tf/target_rotation",target.rotation);\n'
              '      return tfErrors(')
s = s.replace('        auto total = torch::zeros({}, current.xyz.options());',
              '        cdperf::Scope shape_scope("batch_shape_primary_"+std::to_string(main.first.numel())+"_anchor_"+std::to_string(anchor.first.numel()));\n'
              '        auto total = torch::zeros({}, current.xyz.options());', 1)
s = s.replace('          auto total = torch::zeros({}, current.xyz.options());',
              '          cdperf::Scope shape_scope("batch_shape_"+std::to_string(batch.numel()));\n'
              '          auto total = torch::zeros({}, current.xyz.options());', 1)
write('src/change_distill.cpp', s)

s = (S / 'include/frozen_teacher_target_cache.h').read_text(encoding='utf-8')
s = '#include "perf_scope.h"\n' + s
s = s.replace('    torch::NoGradGuard no_grad;',
              '    cdperf::Scope cache_scope("target_cache");\n    torch::NoGradGuard no_grad;')
for expression, label in [
    ('indices.index({~ready_.index_select(0, indices)})', 'cache_missing_indices'),
    ('evaluate(missing).detach()', 'cache_miss_evaluate'),
    ('torch::empty(shape, target.options().requires_grad(false))', 'cache_storage_allocate'),
    ('values_.index_copy_(0, missing, target)', 'cache_write'),
    ('ready_.index_fill_(0, missing, true)', 'cache_ready_update'),
    ('values_.index_select(0, indices)', 'cache_target_gather'),
]:
    s = wrap(s, expression, label)
s = s.replace('    return CD_OP("cache_target_gather", values_.index_select(0, indices));',
              '    cdperf::tensor(cdperf::branch + "/cache_values", values_);\n'
              '    return CD_OP("cache_target_gather", values_.index_select(0, indices));')
write('include/frozen_teacher_target_cache.h', s)

s = (S / 'src/transform_field.cpp').read_text(encoding='utf-8')
s = '#include "perf_scope.h"\n' + s
for expression, label in [
    ('tf_cuda::query(xyz, ratio_tensor, aabbMin(), aabbMax(), gridParameters())', 'hexplane_query'),
    ('tf_cuda::neural(feature, mlpParameters(), mlp_backend)', 'tf_mlp'),
    ('grid_->forward(xyz, ratio_tensor)', 'hexplane_query_legacy'),
    ('torch::nonzero(insideMask(state.xyz)).reshape({-1})', 'network_inside_filter'),
    ('state.select(indices)', 'network_gaussian_gather'),
    ('forward(selected.xyz, ratio)', 'network_residual_forward'),
    ('tf_cuda::output(selected.xyz, selected.log_scaling, selected.rotation,\n        {residual.delta_xyz, residual.delta_log_scale, residual.delta_rotation_raw}, true)', 'output_transform'),
]:
    s = wrap(s, expression, label)
s = s.replace('    // Same-machine complete',
              '    cdperf::tensor("tf/sampled_feature", feature);\n    // Same-machine complete')
s = s.replace('    return {residual[0], residual[1], residual[2]};',
              '    for (size_t i=0;i<residual.size();++i) cdperf::tensor("tf/residual_"+std::to_string(i), residual[i]);\n'
              '    return {residual[0], residual[1], residual[2]};')
write('src/transform_field.cpp', s)

for relative, token, label in [
    ('src/selector_cuda.cpp', '    auto in=tensors.vec();', 'selector_network_forward'),
    ('src/transform_field_cuda.cpp', '    return queryBackward(ctx->get_saved_variables(), gradients.at(0).contiguous());', 'hexplane_backward'),
]:
    s = (S / relative).read_text(encoding='utf-8')
    s = '#include "perf_scope.h"\n' + s
    assert token in s
    s = s.replace(token, '    cdperf::Scope scope("' + label + '");\n' + token, 1)
    if relative.endswith('selector_cuda.cpp'):
        token = '    auto saved=ctx->get_saved_variables();'
        s = s.replace(token, '    cdperf::Scope scope("selector_network_backward");\n' + token)
        token = '    ctx->saved_data["strategy"]=strategy;'
        s = s.replace(token, '    for(size_t i=0;i<stages.size();++i) cdperf::tensor("selector/stage_"+std::to_string(i), stages[i]);\n' + token)
    else:
        s = s.replace('    // Save feature/parameters',
                      '    for(size_t i=0;i<context.size();++i) cdperf::tensor("tf/mlp_stage_"+std::to_string(i),context[i]);\n'
                      '    // Save feature/parameters')
        for token, label in [('    auto saved=ctx->get_saved_variables();', 'tf_mlp_backward'),
                             ('    auto result=outputBackward(', 'output_backward')]:
            s = s.replace(token, '    cdperf::Scope scope("' + label + '");\n' + token, 1)
    write(relative, s)

(D / 'instrumentation.diff').write_text(''.join(changes), encoding='utf-8', newline='\n')
(D / 'raw/instrumentation_manifest.json').write_text(json.dumps({
    'production_sources_modified': False,
    'method': 'expression wrappers and RAII scopes in isolated frozen copies',
    'events': 'same current stream, no synchronization within scopes, resolve after run',
    'native_binary': 'unmodified frozen implementation',
}, indent=2), encoding='utf-8')
print('generated instrumentation; no production files edited')
