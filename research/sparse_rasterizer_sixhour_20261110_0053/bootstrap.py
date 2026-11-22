"""锁定场景分组、输入身份、两个只读基线与六小时墙钟预算。"""
from pathlib import Path
import csv,hashlib,json,time

HERE=Path(__file__).resolve().parent
PREVIOUS=Path('/home/prmi/zwj/experiments/sparse_rasterizer_balanced_20261110')
SAVED=Path('/home/prmi/zwj/realLOD/saved_experiments')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    assert not (HERE/'reports/PROTOCOL.json').exists()
    scenes=json.loads(Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/reports/SCENES.json').read_text())
    external=['TUM_RGBD/freiburg1_desk/rgbd','TUM_RGBD/freiburg1_room/rgbd',
        'TUM_RGBD/freiburg1_xyz/rgbd','TUM_RGBD/freiburg2_xyz/mono',
        'VECtor/corner-slow/mono','VECtor/corner-slow/stereo','VECtor/corridors-dolly/stereo','VECtor/robot-normal/mono']
    for group in ['dual_view_li_best_full_20261107','dual_view_li_vector_best_full_20261107']:
        root=SAVED/group
        rows=list(csv.DictReader((root/'reports/LOD_METRICS.csv').open()))
        for scene_id in external:
            row=next((r for r in rows if r['id']==scene_id),None)
            if row is None:continue
            dataset,scene,mode=scene_id.split('/')
            checkpoints=list((root/'runs'/scene_id).glob('attempt_*/model/*_shutdown/ply/selector.pt'))
            assert len(checkpoints)==1,(scene_id,checkpoints)
            checkpoint=checkpoints[0].parent
            data=Path('/home/prmi/datasets/TUM_RGBD')/('rgbd_dataset_'+scene) if dataset=='TUM_RGBD' else SAVED/'multi_dataset_full_tail_prune1_seed0_20261103/datasets/VECtor'/scene
            config=root/'configs'/scene_id/(mode+'_camera.yaml')
            scenes.append(dict(id=scene_id,dataset_kind=dataset,scene=scene,mode=mode,slug=scene_id.replace('/','__'),
                checkpoint=str(checkpoint),dataset=str(data),camera_config=str(config),
                trajectory=str(checkpoint.parents[1]/'CameraTrajectory_TUM.txt'),gaussians=int(row['gaussian_count'])))
    tune={'Replica/office1/mono','Replica/office2/rgbd','Replica/room0/rgbd','TUM_RGBD/freiburg1_desk/rgbd','VECtor/corner-slow/mono'}
    for scene in scenes:
        scene['split']='tune' if scene['id'] in tune else 'heldout'
        for key in ['checkpoint','dataset','trajectory']:assert Path(scene[key]).exists(),(scene['id'],key,scene[key])
    assert len(scenes)==24 and len({s['id'] for s in scenes})==24
    A=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')
    B=PREVIOUS/'build/sparse_rasterizer_balanced_v1.so'
    before=json.loads((PREVIOUS/'reports/formal/PROTECTED_BEFORE.json').read_text())
    for path,value in before.items():assert Path(path).is_file() and digest(path)==value,path
    for scene in scenes:
        for p in Path(scene['checkpoint']).rglob('*'):
            if p.is_file():before[str(p)]=digest(p)
        for key in ['trajectory','camera_config']:
            p=Path(scene[key]);assert p.is_file(),p;before[str(p)]=digest(p)
    for p in [A,B,Path('/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_20261109/evaluation_runtime/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')]:before[str(p)]=digest(p)
    protocol=dict(start_utc='2026-11-09T16:53:10Z',audit_deadline_utc='2026-11-09T17:38:10Z',
        freeze_target_utc='2026-11-09T21:38:10Z',deadline_utc='2026-11-09T22:53:10Z',wall_budget_s=21600,
        baseline_A=dict(path=str(A),sha256=digest(A)),baseline_B=dict(path=str(B),sha256=digest(B),mode=4),
        original_entry_matches_running_viewer=digest(A)==digest(Path('/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_20261109/evaluation_runtime/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')),
        Full_hard_gate='每场景两轮配对中位延迟不得慢于A，另报p95和重复波动',
        recommendation_gate='推荐最终路径还须每场景两轮Full中位不慢于B；未通过则保留B并单列候选',
        correctness='RGB及radii逐位一致；存在差异的候选不得合并为最终方案',
        tuning_scenes=sorted(tune),heldout_rule='本轮不在heldout场景测候选性能或调参，冻结后一次性复验；此前历史基线结果已公开，不能称从未见过这些场景',
        tune_motion=dict(continuous_fraction=.12,continuous_frames=12,jump_fractions=[.07,.26,.48,.86]),
        final_motion=dict(continuous_fraction=.64,continuous_frames=12,jump_fractions=[.03,.19,.39,.51,.81,.93]),
        local_ratios=[.01,.03,.05,.08,.1,.3,.7],global_ratios=[.01,.05,.1],fixed_requests=[1000,5000],
        diagnostics='计时外采集K/radii可见数/实际Alpha参与点数/Tile实例/成员分布/Kernel/同步/显存',
        unchanged=['selection','actual_K','compensation','alpha_order','stable_depth_order','resolution','SH_degree','FP32','training'],
        no_K_or_ratio_bypass=True,max_bitset_MiB=256,max_total_workspace_MiB=1024,commit_push_deploy=False,
        source_head='26c355b32bf373a75877fc4605a4e41b8de9b47f',created_at=time.time())
    (HERE/'reports/SCENES.json').write_text(json.dumps(scenes,indent=2))
    (HERE/'reports/PROTECTED_BEFORE.json').write_text(json.dumps(before,indent=2))
    (HERE/'reports/PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    print('BOOTSTRAP_COMPLETE',len(scenes),len(before),flush=True)


if __name__=='__main__':main()
