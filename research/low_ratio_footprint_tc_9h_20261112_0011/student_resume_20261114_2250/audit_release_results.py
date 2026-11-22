"""离线完整性、画质和性能归因；保持全部慢例、反序及测量干扰记录。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'
PRIMARY = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]


def geometric(values):
    return float(np.exp(np.log(values).mean()))


def main():
    chosen = json.loads((R / 'SELECTED_RASTER_20261115.json').read_text())
    prefix = 'TAIL' if chosen['mode'] == 203 else 'STRONG'
    pure = pd.read_csv(R / 'RELEASE_THIN_SCENE_RATIO.csv')
    final = pure[pure.variant.eq('new_thin')]
    assert final.groupby('workload').scene.nunique().to_dict() == {'nested': 32, 'student_lod': 18}
    assert set(final['round']) == {1, 2} and final.condition.nunique() == 14
    pure_summary = []
    for (workload, condition), frame in final.groupby(['workload', 'condition']):
        pure_summary.append(dict(workload=workload, condition=condition, scenes=frame.scene.nunique(),
            rounds=frame['round'].nunique(), K=frame.K.median(), p50_us=frame.p50_ms.mean() * 1000,
            p95_us=frame.p95_ms.mean() * 1000, F2_speedup=geometric(frame.F2_speedup),
            previous182_speedup=geometric(frame.previous182_speedup),
            thin182_speedup=geometric(frame.thin182_speedup),
            slower_F2=int(frame.F2_speedup.lt(1).sum()), slower_182=int(frame.previous182_speedup.lt(1).sum()),
            slower_thin182=int(frame.thin182_speedup.lt(1).sum()),
            worst_F2_speedup=frame.F2_speedup.min(), worst_182_speedup=frame.previous182_speedup.min(),
            worst_thin182_speedup=frame.thin182_speedup.min()))
    pd.DataFrame(pure_summary).to_csv(R / 'FINAL_RASTER_BY_CONDITION.csv', index=False)
    final[final.condition.eq('full')].to_csv(R / 'FINAL_RASTER_ALL_FULL.csv', index=False)
    final[(final.F2_speedup < 1) | (final.thin182_speedup < 1)].to_csv(R / 'FINAL_RASTER_ALL_SLOW_CASES.csv', index=False)
    conditions = pd.read_csv(R / (prefix + '_QUALITY_CONDITIONS.csv'))
    quality = conditions[conditions.variant.eq(chosen['variant'])]
    assert quality.psnr_loss.max() <= .5 and quality.groupby('workload').psnr_loss.mean().max() <= .2
    quality.to_csv(R / 'FINAL_RASTER_QUALITY.csv', index=False)
    joint = pd.read_csv(R / 'RELEASE_JOINT_SCENE_RATIO.csv')
    assert joint.scene.nunique() == 18 and set(joint['round']) == {1, 2} and joint.condition.nunique() == 14
    joint_summary = []
    for (variant, condition), frame in joint.groupby(['variant', 'condition']):
        joint_summary.append(dict(variant=variant, condition=condition, scenes=frame.scene.nunique(),
            p50_us=frame.p50_ms.mean() * 1000, p95_us=frame.p95_ms.mean() * 1000,
            A_speedup=geometric(frame.A_speedup), C_speedup=geometric(frame.C_speedup),
            I_speedup=geometric(frame.I_speedup), prepare_speedup=geometric(frame.new_prepare_speedup),
            slower_A=int(frame.A_speedup.lt(1).sum()), slower_I=int(frame.I_speedup.lt(1).sum())))
    pd.DataFrame(joint_summary).to_csv(R / 'FINAL_JOINT_BY_CONDITION.csv', index=False)
    joint_quality = pd.read_csv(R / 'RELEASE_JOINT_QUALITY.csv')
    q = joint_quality[joint_quality.variant.eq('D_final')]
    joint_pass = bool(q.psnr_loss.mean() <= .2 and q.psnr_loss.max() <= .5)
    per = pd.read_csv(R / 'RELEASE_JOINT_PER_FRAME.csv.gz')
    keys = ['scene', 'condition', 'round', 'frame', 'motion']
    controls = per[per.variant.eq('I_previous')][keys + ['T']].rename(columns={'T': 'I_ms'})
    paired = per[per.variant.eq('D_final')].merge(controls, on=keys, validate='one_to_one')
    paired['I_speedup'] = paired.I_ms / paired['T']
    paired[paired.I_speedup.lt(1)].to_csv(R / 'FINAL_JOINT_FRAME_SLOW_CASES.csv.gz', index=False, compression='gzip')
    paired.to_csv(R / 'FINAL_JOINT_PAIRED_FRAMES.csv.gz', index=False, compression='gzip')
    ablations = joint[joint.condition.isin(PRIMARY)].pivot(index=['scene', 'condition', 'round'], columns='variant', values='p50_ms')
    rows = []
    for disabled in ['no_direct', 'no_grouped', 'no_native_full']:
        ratio = ablations[disabled] / ablations.D_final
        rows.append(dict(disabled=disabled, scope='primary_ratio', speedup=geometric(ratio),
            mean_saved_us=((ablations[disabled] - ablations.D_final) * 1000).mean(), slow_cases=int(ratio.lt(1).sum())))
    full = joint[joint.condition.eq('full')].pivot(index=['scene', 'round'], columns='variant', values='p50_ms')
    ratio = full.no_native_full / full.D_final
    rows.append(dict(disabled='no_native_full', scope='full', speedup=geometric(ratio),
        mean_saved_us=((full.no_native_full - full.D_final) * 1000).mean(), slow_cases=int(ratio.lt(1).sum())))
    pd.DataFrame(rows).to_csv(R / 'FINAL_COMPONENT_ABLATIONS.csv', index=False)
    interference = []
    for controller in ['RELEASE_RENDER_20261115', 'RELEASE_JOINT_20261115']:
        filename = controller + ('_MERGED.json' if controller == 'RELEASE_JOINT_20261115' else '_CONTROLLER.json')
        state = json.loads((R / filename).read_text())
        assert state['state'] == 'complete'
        for decision in ['accepted', 'rejected']:
            for name in state[decision]:
                path = R / 'jobs' / (name + '.json')
                data = json.loads(path.read_text())
                assert data['returncode'] == 0
                if decision == 'accepted':
                    assert data['measurement_valid']
                interference.append(dict(controller=controller, decision=decision, **data))
    pd.DataFrame(interference).drop(columns=['command'], errors='ignore').to_csv(R / 'FINAL_JOB_AUDIT.csv', index=False)
    result = dict(passed=joint_pass, selected_raster=chosen['variant'], raster_groups=50, student_groups=18,
        raster_quality_mean=quality.groupby('workload').psnr_loss.mean().to_dict(),
        raster_quality_worst_condition=float(quality.psnr_loss.max()), joint_quality_mean=float(q.psnr_loss.mean()),
        joint_quality_worst_condition=float(q.psnr_loss.max()), joint_quality_worst_frame=float(q.worst_frame_loss.max()),
        joint_quality_pass=joint_pass, joint_id_overlap=float(q.id_overlap.mean()),
        joint_worst_id_overlap=float(q.worst_id_overlap.min()), accepted=sum(x['decision'] == 'accepted' for x in interference),
        rejected=sum(x['decision'] == 'rejected' for x in interference))
    (R / 'FINAL_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
