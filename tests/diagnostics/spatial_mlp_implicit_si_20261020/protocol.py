from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
reg=dict(scenes=['fr1_room','fr1_xyz','fr2_xyz'],ratios=[.01,.05,.1,.15],seeds=[2026,2027],steps=2000,
    tensor_checkpoints=[0,250,1000,2000],hard_steps=sorted(set([0,250]+list(range(100,2001,100)))),
    architecture=[3,32,32,1],activation='SiLU',dtype='FP64 MLP/input/base/score; original FP32 renderer',parameters=1217,
    initialization='LibTorch default Linear hidden initialization; final weight/bias zero',
    normalization='frozen TF grid.aabb_min/max, 2*(xyz-min)/(max-min)-1; no fit/clamp; out-of-domain rows retained/extrapolated',
    input='normalized XYZ ONLY',score='log(RawSI+1e-12)+MLP(normalizedXYZ)',
    selection='exact legacy PG-ST cached uniform,42-iteration detached-offset expected-K calibration,temp1,compact selected opacity ST',
    RNG='same manual_seed before FrozenScene; cached_uniform drawn BEFORE MLP construction; same Bank mt19937 schedule; model init after cached uniform',
    loss='.8 L1+.2(1-fast_ssim), same production implementation',optimizer='Adam theta only, betas=.9/.999 eps=1e-8 weight_decay=0',
    LR_sanity=dict(scene='fr1_xyz',ratio=.05,seed=2026,lrs=[.0001,.0003,.001],steps=250,
        criterion='finite/nonfrozen parameters; smallest final full-Bank deterministic hard photo at step250 (tie: smaller LR); no baseline/eval/C/Fixed accessed'),
    output_update_steps=[250,500,750,1000,1250,1500,1750,2000],output_numeric_epsilon=1e-12,
    direct_groups='never=count0; rarely=1..1000; frequently=>1000 of2000 steps (descriptive >half), no training use',
    smoothness=dict(pairs=100000,seed=81020,bins='scene-specific pair-distance quartiles in normalized Euclidean distance; frozen identical pairs for MLP and PG'),
    formal_models=24,training_Bank_only=True,no_eval_tuning=True,no_regularization=True,no_grid=True,no_PL=True,no_capacity_sweep=True,
    commit=False,stage=False,push=False)
(ROOT/'preregistration.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
