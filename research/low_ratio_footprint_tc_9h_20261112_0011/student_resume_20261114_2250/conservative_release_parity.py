"""跨18组确认保守新调度与起点182联合输出逐位一致，不只比较指标。"""
from release_joint import *


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = HERE / 'reports' / args.output
    out.mkdir(exist_ok=False)
    rows = []
    for spec in student_specs():
        source, student, _ = load_case(spec)
        factory = NativeCameraPreparer(p.MiniCam)
        old = FinalStudentRenderer(source, student, 182, reuse=True, fused=True, finish=True, members=15)
        new = JointReleaseRenderer(source, student, 182, reuse=True, fused=True, finish=True)
        n = len(source.scene.views)
        views = motions(source) + [('continuous_unseen', int(.58 * n)), ('continuous_unseen', int(.58 * n) + 1),
            ('jump_unseen', int(.23 * n)), ('jump_unseen', int(.81 * n))]
        for condition in CONDITIONS:
            for motion, frame in views:
                expected = old.render_from_source(source, frame, source.scene.background, condition, factory)
                actual = new.render_from_source(source, frame, source.scene.background, condition, factory)
                row = dict(scene=spec['id'], condition=condition, motion=motion, frame=frame, K=len(actual[2]),
                    rgb_bitwise=bits_equal(expected[0], actual[0]), radii_bitwise=bits_equal(expected[1], actual[1]),
                    ids_bitwise=torch.equal(expected[2], actual[2]))
                rows.append(row)
                assert row['rgb_bitwise'] and row['radii_bitwise'] and row['ids_bitwise'], row
        del source, student, old, new, expected, actual
        gc.collect()
        torch.cuda.empty_cache()
        print('CONSERVATIVE_PARITY', spec['id'], flush=True)
    append(out / 'checks.csv', rows)
    save(out / 'DONE.json', dict(passed=True, scenes=18, conditions=14, views=14, checks=len(rows),
        scope='new_prepare_182 vs I_previous', bitwise=True, no_images=True))


if __name__ == '__main__':
    main()
