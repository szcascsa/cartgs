"""最终因果判断所需的紧凑证据，不据此启动新训练。"""
import json
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent
q=pd.read_csv(ROOT/'quality_macro.csv');q=q[q.split=='test']
main=q[(q.ratios=='seven')&q.tier.isin(['confirm_continue','mechanism_continue','frozen_mechanism'])]
print('QUALITY',main.groupby(['tier','scene','config'])[['psnr','ssim','lpips','photo']].mean().round(6).to_string())
print('REFERENCE',pd.read_csv(ROOT/'references_per_ratio.csv').query("split=='test'").groupby(['scene','reference'])[['psnr','ssim','lpips','photo']].mean().round(6).to_string())
print('RATIO_SUPPORT',pd.read_csv(ROOT/'per_gaussian_ratio_support.csv').to_string(index=False))
d=pd.read_csv(ROOT/'ratio_support_counterfactual_per_view.csv');d=d[d.split=='test'];print('SUPPORT_ABLATION',d.groupby(['scene','ratio','ablation'])[['psnr','photo']].mean().round(6).to_string())
ab=pd.read_csv(ROOT/'mechanism_ablation_summary.csv');ab=ab[(ab.model=='FrozenBaseHead8')&(ab.split=='test')]
print('FEATURE_ABLATION',ab.groupby(['scene','ablation']).psnr_delta.mean().round(6).to_string())
feat=pd.read_csv(ROOT/'feature_statistics.csv');feat=feat[(feat.model=='FrozenBaseHead8')&(feat.tier.isin(['mechanism_continue','frozen_mechanism']))];print('FEATURE_STATS',feat.groupby(['scene','bin'])[['anchors','visited','gradient_covered','feature_rms','effective_rank','sample_visits_mean','actual_adam_row_updates_mean']].mean().round(4).to_string())
print('PERFORMANCE',pd.read_csv(ROOT/'performance_summary.csv').round(6).to_string(index=False))
p=pd.read_csv(ROOT/'physical_output_changes.csv');p=p[(p.model=='FrozenBaseHead8')&(p.bin=='all')];print('PHYSICAL_CHANGE',p.groupby(['scene','metric'])[['median','p90','maximum']].mean().round(6).to_string())
