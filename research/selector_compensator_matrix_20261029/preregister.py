"""锁定48组ID与预先声明的质量/几何/开销判据。"""
from pathlib import Path
import json,shutil
R=Path(__file__).resolve().parent;REPO=R.parents[1]
for folder,name in [('anchor128_online_office2_20261029','ANCHOR128_ONLINE_OFFICE2_REPORT.md'),('anchor128_offset_online_office2_20261029','ANCHOR128_OFFSET_ONLINE_OFFICE2_REPORT.md'),('anchor128_offset_bound2_office2_20261029','ANCHOR128_OFFSET_BOUND2_REPORT.md')]:shutil.copy2(REPO/'tests/diagnostics'/folder/name,R/'evidence'/name)
shutil.copy2(REPO/'tests/diagnostics/anchor128_offset_online_office2_20261029/anchor128.yaml',R/'baseline25.yaml')
rows=[]
for cap in [2,0]:
    for ai,arch in enumerate(['R0','R1','E1','E2','E3','E4']):
        for gi,(sg,cg) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
            idx=1+ai*4+gi+(0 if cap==2 else 24);E=arch.startswith('E');sf=arch in ['E2','E4'];cf=arch in ['R1','E3','E4']
            rows.append(dict(id=idx,architecture=arch,global_mode=f'G{sg}{cg}',bound=cap,selector_feature=E,selector_fourier=sf,compensator_fourier=cf,selector_global=sg,compensator_global=cg,selector_dim=(6+(384 if sf else 128) if E else 11)+3*sg,compensator_dim=6+(384 if cf else 128)+3*cg,seed=0))
assert len(rows)==48 and sorted(x['id'] for x in rows)==list(range(1,49))
(R/'matrix.json').write_text(json.dumps(rows,indent=2))
protocol=dict(baseline_id=25,ratios=[.01,.05,.1,.2,.3,.4],unseen=[.025,.075,.15,.25,.35],scene='Replica office2 RGB-D',mapping_target=13400,online_cap=False,forced_tail=False,lod_resolution=[1200,680],tail='full_then_joint_lod',seeds=[0,1],
    qualification=dict(psnr_noninferiority_margin_db=.20,required_tiers=['train6_mean','unseen5_mean','1%','5%','Full'],paired_seed_rule='both optimization seeds pass each tier; paired-view bootstrap lower95% >= -.20; not independent-map inference',ssim_margin=.005,lpips_margin=.01,baseline_noise_rule='if baseline seed variation>.20dB or budget imbalance>10% Master or>15% Full, mark unresolved, do not relax margins',geometry_rule='1% and5% displacement p90<=.60baseline, p99<=baseline; effective visible opacity>=1/255 Scale p99<=1.25baseline',compute_rule='matched N/K Selector+Compensation forward/backward median<=1.25baseline; online wall and update-normalized cost both reported; shared GPU timing limitations explicit'),
    calibration='baseline25 retains original first-compensation normalization; fixed buffers from its final checkpoint reused as training-only calibration by other seed0 variants, including before3000 for E/global; seed1 candidates use seed1 baseline buffers',
    evaluation='common covered RGB identities; uniform up to64 outside all Bank union for primary nonBank; ownBank uniform up to32 per model separately; same-K=floor(r*N25), insufficient eligible is unreachable, never lower K globally',
    preservation='server main read only; independent source/build/bin/lib/runs/evaluation/reports; dependencies read only; no push/default switch')
(R/'preregistration.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'SOURCE_BASELINE_AUDIT.md').write_text('''# 当前源码与正式baseline核查

25号R0-G00-B0为正式baseline。Compensator为共享Anchor128、asinh及固定标准化后的局部offset3、连续log-ratio/sin/cos3，共134D；ratio门控、64/64主干、四Head、14D变换及Joint不改。Selector为线性局部offset3、激活Scale3、SH DC3、激活Opacity1、log-ratio1，共11D，11→32→ReLU→32→ReLU→2。

生产默认仍是spatial_localxyz；baseline配置明确选spatial_anchor128_offset_gate。Bound2及历史139D分支不作为25号。E系列移除Selector的Scale/DC/Opacity，使用与Compensator相同预处理的offset3、ratio3、原始共享Feature，区别于历史139D。

快照仅纳入相关11个源码改动、CMake引用的4个小测试源、本研究配置/协议/小报告。模型、数据、构建、密码及大型快照不纳入，其他未跟踪工作保留。研究实现仅修改独立副本。

固定统计时序：25号保持原3000步后首次补偿初始化；其余seed0变体使用25号固定缓冲作为训练来源校准，使E首次LoD可用，两分支共用统计。统计来自训练地图，未按nonBank质量选择。与历史各自估计统计的差异单列。配对seed1同理先跑25号再校准候选。
''',encoding='utf-8')
print('MATRIX_PREREGISTERED',len(rows))
