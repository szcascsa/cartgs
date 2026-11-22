"""把计时外诊断绑定到同输入最终轻量入口；不以名义Ratio代替实际K。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def main():
    selected = json.loads((R / 'SELECTED_RASTER_20261115.json').read_text())
    prefix = 'TAIL' if selected['mode'] == 203 else 'STRONG'
    counters = pd.read_csv(R / (prefix + '_FINAL_WORKLOADS.csv.gz'))
    mapping = {'F2': 'F2', 'previous182': 'thin182', selected['variant']: 'new_thin'}
    counters = counters[counters.variant.isin(mapping)].copy()
    counters['variant'] = counters.variant.map(mapping)
    keys = ['workload', 'scene', 'condition', 'frame', 'variant']
    assert not counters.duplicated(keys).any()
    performance = pd.read_csv(R / 'RELEASE_THIN_PER_FRAME.csv.gz')
    performance = performance[performance.variant.isin(mapping.values())]
    fields = [name for name in counters.columns if name not in ['job', 'N', 'K']]
    joined = performance.merge(counters[fields], on=keys, validate='many_to_one')
    assert len(joined) == len(performance)
    joined['interactions_per_input'] = joined.alpha_interactions / joined.K.clip(lower=1)
    joined['instances_per_input'] = joined.tile_instances / joined.K.clip(lower=1)
    joined.to_csv(R / 'FINAL_WORKLOAD_TIMING.csv.gz', index=False, compression='gzip')
    joined.groupby(['workload', 'condition', 'variant']).agg(K=('K', 'median'),
        preprocess_visible=('preprocess_visible', 'median'), alpha_participating=('alpha_participating', 'median'),
        alpha_interactions=('alpha_interactions', 'median'), tile_instances=('tile_instances', 'median'),
        active_tiles=('active_tiles', 'median'), members_p50=('members_p50', 'median'),
        members_p95=('members_p95', 'median'), members_max=('members_max', 'max'),
        overflow_tiles=('overflow_tiles', 'sum'), interactions_per_input=('interactions_per_input', 'median'),
        T_us=('T', lambda x: x.median() * 1000)).reset_index().to_csv(R / 'FINAL_WORKLOAD_BY_CONDITION.csv', index=False)
    primary = joined[joined.condition.isin([f'local{x:g}' for x in [.1, .2, .3, .4, .5]])]
    matching = ['workload', 'scene', 'condition', 'frame', 'round']
    before = primary[primary.variant.eq('thin182')][matching + ['alpha_interactions', 'tile_instances', 'T']]
    after = primary[primary.variant.eq('new_thin')].merge(before, on=matching, suffixes=('', '_old'), validate='one_to_one')
    after['interaction_fraction'] = after.alpha_interactions / after.alpha_interactions_old.clip(lower=1)
    after['instance_fraction'] = after.tile_instances / after.tile_instances_old.clip(lower=1)
    after['speedup'] = after.T_old / after['T']
    after.groupby('workload').agg(interaction_fraction=('interaction_fraction', 'median'),
        instance_fraction=('instance_fraction', 'median'), speedup=('speedup', lambda x: np.exp(np.log(x).mean())),
        matching_frames=('speedup', 'size')).to_csv(R / 'FINAL_WORKLOAD_INCREMENT.csv')
    print('WORKLOAD_ATTRIBUTION_COMPLETE', len(joined))


if __name__ == '__main__':
    main()
