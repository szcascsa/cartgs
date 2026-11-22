"""真实教师与完整轨迹的准入检查，不更新模型。"""
from scene_runtime import *
from compensation_student import StudentCompensator, teacher_fingerprint
from distill_compensation import student_residual, distillation_loss, view_split
from renderer import render_prepared


def main():
    slug = sys.argv[1]
    scene, source = load_scene(slug)
    assert int(source.runtime._base.joint_schema) == 2
    assert source.runtime.total == scene['gaussians']
    student = StudentCompensator.from_teacher(source.runtime._base.spatial)
    assert sum(p.numel() for p in student.parameters()) == 3534
    train, holdout = view_split(len(source.scene.views))
    validation = [holdout[int(i)] for i in np.linspace(0,len(holdout)-1,3,dtype=int)]
    pool = [v for v in holdout if v not in validation]
    test = [pool[int(i)] for i in np.linspace(0,len(pool)-1,32,dtype=int)]
    assert not (set(train)&set(test) or set(train)&set(validation) or set(test)&set(validation))
    rows = []
    for frame in validation:
        gt = source.scene.ground_truth(frame)
        assert gt is not None and gt.shape == (3, source.scene.calibration['height'], source.scene.calibration['width'])
        for ratio in [.01, .05, .1, .3, .5, .7]:
            for local in source.local_modes:
                case = source.sample(frame, ratio, local)
                spatial = source.runtime._base.spatial
                encoded = spatial.encode(case['inputs']) + spatial.ratio_projection(case['ratio_code'])
                if case['condition'] is not None:
                    encoded = encoded + spatial.view_projection(case['condition'])
                hidden = torch.relu(spatial.body(torch.relu(encoded)))
                raw = torch.cat([head(hidden) for head in spatial.heads.values()],dim=1)
                expected = torch.cat((raw[:,:3]*case['reference'],raw[:,3:]),dim=1)
                assert torch.allclose(expected,case['target'],rtol=1e-4,atol=1e-5),(scene['id'],frame,ratio,local)
                image = render_prepared(case['camera'],case['packet'],source.scene.background)['render']
                assert torch.isfinite(image).all()
                rows.append(dict(frame=frame, ratio=ratio, local=local, K=len(case['ids']),
                    mse=float((image.clamp(0,1)-gt).square().mean()),
                    teacher_xyz_max=float((case['target'][:,:3].norm(dim=1)/case['reference'].flatten()).max()),
                    teacher_cuda_max_abs=float((expected-case['target']).abs().max()),
                    scene_n=scene['gaussians']))
    generator = torch.Generator().manual_seed(0)
    cases = [source.sample(train[(i*37)%len(train)], [.01,.05,.1,.3,.5,.7][i//2], bool(i%2), 1024, generator)
        for i in range(12)]
    student.calibrate(cases)
    value = sum(distillation_loss(student,cases[0],student_residual(student,cases[0])).values())
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(p.grad is None for p in source.runtime._base.model.parameters())
    assert source.fingerprint == teacher_fingerprint(source.runtime._base.model, source.runtime._base.pc)
    metadata = dict(passed=True, scene=scene, fingerprint=source.fingerprint,
        full_trajectory_frames=len(source.scene.views), training_views=len(train),
        validation_views=validation, test_views=test, parameters=3534,
        calibration_loss=float(value), master_gradients_absent=True, teacher_gradients_absent=True,
        calibration=source.scene.calibration, keyframe_count=len(json.loads((Path(scene['checkpoint'])/'cameras.json').read_text())),
        source_trace=dict(trajectory_sha256=sha(scene['trajectory']), camera_config_sha256=sha(scene['camera_config'])), rows=rows)
    folder = ROOT/'reports'/slug
    folder.mkdir(exist_ok=True)
    save(folder/'PREFLIGHT.json',metadata)
    save(folder/'FRAME_ORDER.json',[{k:v for k,v in view.items() if k!='pose'} for view in source.scene.views])
    print('PREFLIGHT_PASS',scene['id'],len(source.scene.views),scene['gaussians'],flush=True)


if __name__ == '__main__':
    main()
