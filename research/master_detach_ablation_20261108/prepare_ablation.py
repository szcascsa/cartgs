"""准备32组全分支固定Master消融，只生成源码、配置和待启动队列。"""
from pathlib import Path
from collections import Counter
import copy
import difflib
import hashlib
import json
import shutil
import time

R = Path('/home/prmi/zwj/realLOD/ablation')
B = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
V = B.parent/'dual_view_li_vector_best_full_20261107'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(text, before, after, count=1):
    assert text.count(before) == count, before
    return text.replace(before, after)


def patch_source():
    changes = {}
    for relative in ['src/gaussian_mapper.cpp', 'src/real_lod.cpp',
                     'research/global_view_joint_lod_20261105/lod_from_full.cpp']:
        path = R/'source'/relative
        original = path.read_text()
        text = original
        if relative == 'src/gaussian_mapper.cpp':
            text = replace(text,
                'real_lod_full_loss_weight_ < 0.f ||\n'
                '      (real_lod_full_loss_weight_ > 0.f && selector_detach_gaussian_parameters_)',
                'real_lod_full_loss_weight_ < 0.f')
            text = replace(text,
                'RealLoD.full_loss_weight must be nonnegative; Full supervision requires a live Master',
                'RealLoD.full_loss_weight must be finite and nonnegative')
            text = replace(text,
                'real_lod_.enabled&&real_lod_full_loss_weight_>0&&!selector_detach_gaussian_parameters_&&',
                'real_lod_.enabled&&real_lod_full_loss_weight_>0&&', count=2)
        elif relative == 'src/real_lod.cpp':
            text = replace(text,
                'real_lod_full_loss_weight_ > 0.f &&\n              !selector_detach_gaussian_parameters_',
                'real_lod_full_loss_weight_ > 0.f')
            text = replace(text, 'Full Fine supervision requires a live Master',
                'Full Fine supervision requires an active Fine phase and positive weight')
            text = replace(text,
                'background_,override_color_,1.f,false,{},false,collect_view||collect_li)',
                'background_,override_color_,1.f,false,{},selector_detach_gaussian_parameters_,collect_view||collect_li)')
            text = replace(text,
                'full_loss+=opt_params_.opacity_reg_*gaussians_->getOpacityActivation().abs().mean();',
                'full_loss+=opt_params_.opacity_reg_*(selector_detach_gaussian_parameters_?\n'
                '          gaussians_->getOpacityActivation().detach():gaussians_->getOpacityActivation()).abs().mean();')
        else:
            text = replace(text,
                'real_lod_.enabled&&!selector_detach_gaussian_parameters_&&real_lod_full_loss_weight_>0',
                'real_lod_.enabled&&selector_detach_gaussian_parameters_&&real_lod_full_loss_weight_>0')
            text = replace(text, 'tensor.requires_grad_(true);', 'tensor.requires_grad_(false);')
            text = replace(text,
                'auto initial_master=gaussians_->xyz_.detach().clone();',
                'std::vector<torch::Tensor> initial_master;\n'
                '    const std::vector<torch::Tensor> master={gaussians_->xyz_,gaussians_->features_dc_,\n'
                '        gaussians_->features_rest_,gaussians_->opacity_,gaussians_->scaling_,gaussians_->rotation_};\n'
                '    for(const auto& tensor:master)initial_master.push_back(tensor.detach().clone());')
            text = replace(text,
                'flexgs_lod_master_updates_==real_lod_.lod_steps&&flexgs_lod_selector_updates_==real_lod_.lod_steps',
                'flexgs_lod_master_updates_==0&&flexgs_lod_selector_updates_==real_lod_.lod_steps')
            text = replace(text,
                'const auto final=result_dir_/(std::to_string(getIteration())+"_shutdown");',
                'for(size_t i=0;i<master.size();++i) {\n'
                '      TORCH_CHECK(!master[i].requires_grad()&&!master[i].grad().defined(),"Detached Master received gradient");\n'
                '      TORCH_CHECK(master[i].view(torch::kInt32).equal(initial_master[i].view(torch::kInt32)),\n'
                '                  "Frozen Master changed in field ",i);\n'
                '      TORCH_CHECK(gaussians_->optimizer_->state().find(master[i].unsafeGetTensorImpl())==\n'
                '                  gaussians_->optimizer_->state().end(),"Frozen Master created Adam state");\n'
                '    }\n'
                '    for(const auto& table:selector_network_->anchors()->compensationFeatureTables()) {\n'
                '      auto state=gaussians_->optimizer_->state().find(table.unsafeGetTensorImpl());\n'
                '      TORCH_CHECK(state!=gaussians_->optimizer_->state().end(),"Anchor Feature missing Adam state");\n'
                '    }\n'
                '    const auto final=result_dir_/(std::to_string(getIteration())+"_shutdown");')
            text = replace(text,
                'report["master_xyz_changed"]=!torch::equal(initial_master,gaussians_->xyz_.detach());',
                'report["master_xyz_changed"]=false;\n'
                '    report["all_six_master_fields_bitwise_frozen"]=true;\n'
                '    report["master_gradients_absent"]=true;report["master_adam_state_absent"]=true;\n'
                '    report["anchor_feature_adam_present"]=true;')
        path.write_text(text)
        changes[relative] = dict(before_sha256=sha(B/'source'/relative), after_sha256=sha(path))
        with (R/'reports/MASTER_DETACH.diff').open('a') as stream:
            stream.writelines(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                                 fromfile='baseline/'+relative, tofile='ablation/'+relative))
    write(R/'reports/SOURCE_CHANGES.json', changes)


def main():
    assert R.is_dir() and not (R/'LOD_PIPELINE.json').exists()
    for name in ['configs', 'reports', 'logs', 'runs', 'evaluation', 'validation', 'objects']:
        (R/name).mkdir(exist_ok=True)
    (R/'HOLD').write_text('等待用户明确授权启动，禁止训练及场景smoke\n')
    for name in ['source', 'runtime', 'evaluation_runtime']:
        shutil.copytree(B/name, R/name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    helpers = ['pipeline.py', 'oom_retry.py', 'lod_eval_entry.py', 'eval_joint_entry.py',
               'verify_initialization.py', 'validate_startup.py', 'test_queue.py', 'supervisor.py']
    for name in helpers:
        shutil.copy2(B/name, R/name)
    shutil.copy2(V/'memory_guard.py', R/'memory_guard.py')
    shutil.copy2(V/'configs/corridors_stereo_memory.json', R/'configs/corridors_stereo_memory.json')
    shutil.copy2(B/'configs/vector_stereo_native_maps.yml.gz', R/'configs/vector_stereo_native_maps.yml.gz')
    config = (B/'configs/dual_view_li.yaml').read_text()
    config = replace(config, 'Selector.detach_gaussian_parameters: 0', 'Selector.detach_gaussian_parameters: 1')
    (R/'configs/dual_view_li.yaml').write_text(config)
    write(R/'reports/CONFIG_DIFFERENCE.json', dict(only_changed_key='Selector.detach_gaussian_parameters',
        baseline=0, ablation=1, full_loss_weight=1, li_weight=0.1,
        branches=['Full', 'global_LoD', 'local_LoD'], master_frozen=True))
    states = [read(B/'LOD_PIPELINE.json'), read(V/'LOD_PIPELINE.json')]
    jobs = []
    for state, batch in zip(states, [B, V]):
        assert state['phase'] == 'finished'
        for completed in state['jobs']:
            assert completed['state'] == 'complete'
            job = copy.deepcopy(completed)
            for key in ['attempt_history', 'process', 'attempt', 'trained_model', 'error', 'observed_at',
                        'log_bytes', 'trace_bytes', 'retry_round', 'retry_round_history', 'round_first_attempt',
                        'skipped_unix', 'skip_reason']:
                job.pop(key, None)
            job.update(state='pending', evaluations=[], attempt_history=[], health_failures=0, oom_failures=0,
                       baseline_model=completed['trained_model'], baseline_batch=str(batch.resolve()))
            for key in ['camera_config', 'mapping_config']:
                original = Path(completed[key])
                relative = Path('configs')/job['dataset']/job['scene']/job['mode']/original.name
                output = R/relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, output)
                assert sha(original) == sha(output)
                job[key] = str(output)
            jobs.append(job)
    assert len(jobs) == 32 and Counter(j['dataset'] for j in jobs) == {'Replica': 16, 'TUM_RGBD': 10, 'VECtor': 6}
    runner = (V/'run_batch.py').read_text()
    runner = replace(runner, "assert len(state['jobs'])==6 and all(j['dataset']=='VECtor' for j in state['jobs'])",
        "assert len(state['jobs'])==32 and collections.Counter(j['dataset'] for j in state['jobs'])=={'Replica':16,'TUM_RGBD':10,'VECtor':6}")
    runner = replace(runner, "assert not stage['master_frozen_in_lod'] and not stage['maturity_gate'] and not stage['maturity_loss']",
        "assert stage['master_frozen_in_lod'] and not stage['maturity_gate'] and not stage['maturity_loss']")
    runner = replace(runner, "ply=original_verify(out,target);stage=p.read(ply/'real_lod_state.json')",
        "ply=original_verify(out,target);stage=p.read(ply/'real_lod_state.json')\n"
        "    frozen=p.read(out/'model/training_verified.json')\n"
        "    assert all(frozen[k] for k in ['all_six_master_fields_bitwise_frozen','master_gradients_absent','master_adam_state_absent','anchor_feature_adam_present'])")
    runner = runner.replace('Full＋全局LoD＋局部LoD，LI=0.1；Master可更新、固定拓扑；未见ratio只评价。',
        'Full＋全局LoD＋局部LoD，LI=0.1；三路Master均detach且固定，网络与Anchor Feature训练；未见ratio只评价。')
    runner = runner.replace('6组VECtor双网络真实视角＋LI独立队列',
        '32组Master固定、双网络真实视角＋LI独立队列')
    runner = replace(runner, 'def main():\n    resource.setrlimit',
        "def main():\n    assert not (R/'HOLD').exists() and p.read(R/'START_AUTHORIZED.json')['authorized']\n    resource.setrlimit")
    (R/'run_batch.py').write_text(runner)
    pipeline = (R/'pipeline.py').read_text()
    pipeline = replace(pipeline, "assert result['passed'] and result['topology_unchanged'] and result['master_xyz_changed']",
        "assert result['passed'] and result['topology_unchanged'] and not result['master_xyz_changed']")
    pipeline = replace(pipeline, "'lod_steps', 'master_updates', 'selector_updates', 'full_training_renders',",
        "'lod_steps', 'selector_updates', 'full_training_renders',")
    pipeline = replace(pipeline, "assert stage['lod_nonfinite'] == 0 and sum(stage['ratio_steps']) == n",
        "assert stage['master_updates'] == 0 and stage['master_frozen_in_lod']\n    assert stage['lod_nonfinite'] == 0 and sum(stage['ratio_steps']) == n")
    pipeline = replace(pipeline, 'def main():\n    lock =',
        "def main():\n    assert not (R/'HOLD').exists() and read(R/'START_AUTHORIZED.json')['authorized']\n    lock =")
    pipeline = pipeline.replace('Fine复用原生Full+轮转LoD，每步累积后一次Adam；Master可更新，拓扑固定。',
        'Fine保留Full与轮转LoD；三路Master均固定，网络和Anchor Feature每步共享一次Adam。')
    (R/'pipeline.py').write_text(pipeline)
    supervisor = (R/'supervisor.py').read_text()
    supervisor = replace(supervisor, "def main():\n    lock=", "def main():\n    assert not (R/'HOLD').exists() and p.read(R/'START_AUTHORIZED.json')['authorized']\n    lock=")
    (R/'supervisor.py').write_text(supervisor)
    start = '''"""仅在用户明确说跑后，使用 --start 启动。"""
from pathlib import Path
import argparse
import subprocess
import sys
import time
import run_batch as b
import supervisor

parser=argparse.ArgumentParser()
parser.add_argument('--start',action='store_true',required=True)
parser.parse_args()
r=Path(__file__).resolve().parent
assert b.p.read(r/'reports/PREPARED.json')['passed']
assert b.p.read(r/'reports/BOOTSTRAP.json')['phase']=='build_complete'
for name in ['SUPERVISOR.json','LOD_CONTROLLER.json']:
    assert not supervisor.alive(b.p.read(r/name) if (r/name).exists() else {})
assert not supervisor.running(b.p.read(r/'LOD_PIPELINE.json'))
b.p.write(r/'START_AUTHORIZED.json',dict(authorized=True,started_unix=time.time()))
if (r/'HOLD').exists():(r/'HOLD').unlink()
with (r/'logs/supervisor.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'-u',str(r/'supervisor.py')],cwd=r,env=b.environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(b.p.identity(child.pid))
'''
    (R/'start_queue.py').write_text(start)
    protocol = copy.deepcopy(states[0]['protocol'])
    protocol.update(master_updates=False, master_gradient_scope='Full/global_LoD/local_LoD detached')
    protocol.pop('excluded_dataset', None)
    write(R/'LOD_PIPELINE.json', dict(phase='prepared_held', jobs=jobs, validated_modes=[],
        protocol=protocol, counts={'pending': 32},
        requires_explicit_user_start=True, master_gradient_scope='all_three_branches_detached'))
    protected = read(B/'PROTECTED_BEFORE.json')
    protected['baseline_models'] = {}
    for job in jobs:
        for path in Path(job['baseline_model']).rglob('*'):
            if path.is_file():
                protected['baseline_models'][str(path)] = sha(path)
    write(R/'PROTECTED_BEFORE.json', protected)
    patch_source()
    write(R/'reports/SOURCE_HASHES.json', {str(path.relative_to(R/'source')): sha(path)
        for path in (R/'source').rglob('*') if path.is_file() and not path.is_symlink()
        and not any(parent.is_symlink() for parent in path.parents if parent != R)})
    write(R/'reports/BOOTSTRAP.json', dict(phase='build_pending', training_started=False))
    protocol = ['# Master固定消融队列', '',
        '用户确认Full、全局LoD、局部LoD三路全部detach Master，整个Fine阶段固定六类Gaussian参数。',
        'Selector、Compensator和共享Anchor Feature继续训练，Full渲染和Loss继续计算，Full提供同视图LI。',
        '沿用Full＋全局LoD＋局部LoD、LI=0.1、一次共享Adam、原Loss权重和学习率。',
        'Master参数必须逐位不变、无梯度、无新Adam状态；网络和Feature必须产生更新。',
        'Replica16组、TUM10组、VECtor6组，共32组。源Full赢家和Fine步数逐项绑定baseline。',
        'seed=0；训练ratio=[0.01,0.05,0.1,0.3,0.5,0.7]；未见ratio=[0.03,0.08,0.2,0.4,0.6]。',
        '各组23项完整轨迹评价，保存最终模型和数值，不保存渲染图片。双目沿用左图校正map。',
        'OOM和健康停止各最多额外重试2次。corridors-dolly双目上限29GiB并保留2GiB整机余量保护。',
        'HOLD与启动授权双重检查，等用户明确说跑后启动。各入口6步场景smoke也延后到授权启动之后。', '',
        '|场景／模式|Fine步数|源Full PSNR|', '|---|---:|---:|']
    protocol.extend(f"|{j['id']}|{j['target_steps']}|{j['full_psnr']:.6f}|" for j in jobs)
    (R/'reports/ABLATION_PROTOCOL.md').write_text('\n\n'.join(protocol)+'\n')
    print(json.dumps(dict(prepared=True, groups=len(jobs), steps=sum(j['target_steps'] for j in jobs),
                         counts=dict(Counter(j['dataset'] for j in jobs)), held=True, training_started=False)))


if __name__ == '__main__':
    main()
