"""三组共同帧配对汇总，残差关闭诊断单列，审计真实曝光与历史对照限制。"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')
OUT=ROOT/'evaluation/reports'
TRAIN=[.01,.05,.1,.3,.5,.7];UNSEEN=[.03,.08,.2,.4,.6]
GROUPS=['baseline','view_no_li','view_li01','raw_no_li','raw_li01']
NAMES={'baseline':'原联合无LI','view_no_li':'39D 无LI','view_li01':'39D LI=0.1','raw_no_li':'18D 无LI','raw_li01':'18D LI=0.1'}
COMPARISONS=[('baseline',g) for g in GROUPS[1:]]+[('view_no_li','raw_no_li'),('view_li01','raw_li01'),('raw_no_li','raw_li01')]


def write(name,rows):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def split(values):
    for label in ['all','bank','nonbank']:
        chosen=[r for r in values if label=='all' or (r['bank']=='True')==(label=='bank')]
        assert len(chosen)==dict(all=792,bank=72,nonbank=720)[label]
        yield label,chosen


def stats(values,key,item,prefix=None):
    a=np.array([float(r[key]) for r in values]);assert np.isfinite(a).all()
    prefix=key if prefix is None else prefix
    for label,value in [('mean',a.mean()),('median',np.median(a)),('p10',np.percentile(a,10)),('p90',np.percentile(a,90))]:
        item[prefix+'_'+label]=float(value)


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    with (OUT/'PER_VIEW.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==150480
    cases=defaultdict(list)
    for r in rows:cases[(r['group'],r['case'])].append(r)
    summaries=[]
    for (group,case),values in cases.items():
        for partition,chosen in split(values):
            item={k:values[0][k] for k in ['group','case','scope','ratio','ratio_split','diagnostic']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K','candidate_N','requested_K']:
                stats(chosen,key,item)
            item['budget_shortfall_frames']=sum(int(r['budget_shortfall'])>0 for r in chosen);summaries.append(item)
    write('SUMMARY.csv',summaries);(OUT/'SUMMARY.json').write_text(json.dumps(summaries,indent=2))
    lookup={(r['group'],r['case'],int(r['frame'])):r for r in rows};assert len(lookup)==len(rows)
    pairs=[]
    for reference,group in COMPARISONS:
        for row in rows:
            if row['group']!=group or row['diagnostic']=='True':continue
            ref=lookup[(reference,row['case'],int(row['frame']))]
            assert row['image']==ref['image'] and row['timestamp']==ref['timestamp']
            item={k:row[k] for k in ['case','scope','ratio','ratio_split','frame','bank']}
            item.update(reference=reference,group=group)
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:
                item['delta_'+key]=float(row[key])-float(ref[key])
            item['same_selection']=row['selection_sha256']==ref['selection_sha256'];pairs.append(item)
    write('PAIRED_PER_VIEW.csv',pairs)
    paired_cases=defaultdict(list)
    for r in pairs:paired_cases[(r['reference'],r['group'],r['case'])].append(r)
    paired_summary=[]
    for (reference,group,case),values in paired_cases.items():
        for partition,chosen in split(values):
            item={k:values[0][k] for k in ['reference','group','case','scope','ratio','ratio_split']}
            item.update(partition=partition,frames=len(chosen))
            for key in ['psnr','ssim','lpips','photo_l1','K','contributing_K','projected_K']:stats(chosen,'delta_'+key,item)
            item['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in chosen]))
            item['same_K_fraction']=float(np.mean([r['delta_K']==0 for r in chosen]));paired_summary.append(item)
    write('PAIRED_SUMMARY.csv',paired_summary);(OUT/'PAIRED_SUMMARY.json').write_text(json.dumps(paired_summary,indent=2))
    index={(r['group'],r['case'],r['partition']):r for r in summaries}
    paired_index={(r['reference'],r['group'],r['case'],r['partition']):r for r in paired_summary}
    gates=[]
    for ref,group in COMPARISONS:
        d=lambda scope,r:paired_index[(ref,group,f'{scope}_{r:g}','all')]['delta_psnr_mean']
        g=dict(reference=ref,group=group,global_train_mean_delta=float(np.mean([d('global',r) for r in TRAIN])),
            global_train_min_delta=min(d('global',r) for r in TRAIN),full_delta=paired_index[(ref,group,'full','all')]['delta_psnr_mean'],
            view_train_mean_delta=float(np.mean([d('view',r) for r in TRAIN])),
            view_unseen_mean_delta=float(np.mean([d('view',r) for r in UNSEEN])),
            view_1pct_delta=d('view',.01),view_5pct_delta=d('view',.05))
        g['quality_guard_passed']=g['global_train_mean_delta']>=-.1 and g['global_train_min_delta']>=-.2 and g['full_delta']>=-.1
        gates.append(g)
    (OUT/'QUALITY_GATES.json').write_text(json.dumps(gates,indent=2))

    diagnostic=[]
    for group in GROUPS[1:]:
        for off,on in [(f'view_off_{r:g}',f'view_{r:g}') for r in [.01,.05,.1]]+[(f'points_off_{b}',f'points_{b}') for b in [1000,2000]]:
            values=[]
            for frame in range(792):
                a,b=lookup[(group,off,frame)],lookup[(group,on,frame)]
                assert a['K']==b['K'],'同checkpoint开关残差改变提交K'
                v=dict(bank=a['bank'])
                for key in ['psnr','ssim','lpips','contributing_K']:v['delta_'+key]=float(b[key])-float(a[key])
                values.append(v)
            for partition,chosen in split(values):
                item=dict(group=group,case=on,partition=partition,frames=len(chosen),same_K=True)
                for key in ['psnr','ssim','lpips','contributing_K']:stats(chosen,'delta_'+key,item)
                item['psnr_improved_fraction']=float(np.mean([v['delta_psnr']>0 for v in chosen]));diagnostic.append(item)
    write('RESIDUAL_ON_OFF.csv',diagnostic);(OUT/'RESIDUAL_ON_OFF.json').write_text(json.dumps(diagnostic,indent=2))

    old=Path('/home/prmi/zwj/experiments/view_budget_residual_full_20261105')
    state=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    old_state=json.loads((old/'reports/TRAINING_STATUS.json').read_text())
    training=[];experiments=[]
    for group in GROUPS:
        folder=PREVIOUS/'runs/full_no_li' if group=='baseline' else (ROOT if group.startswith('raw_') else old)/'runs'/f'full_{group}'
        with (folder/'real_lod_steps.csv').open() as f:steps=list(csv.DictReader(f))
        local=[json.loads(s) for s in (folder/'view_budget_steps.jsonl').read_text().splitlines()]
        fine=json.loads((folder/'23560_shutdown/ply/real_lod_state.json').read_text())
        assert len(steps)==len(local)==11580 and all(v['adam_commits_this_step']==1 and not v['strict_topk'] for v in local)
        elapsed=(state if group.startswith('raw_') else old_state)['groups'][group]['seconds']
        item=dict(group=group,reused=not group.startswith('raw_'),seconds=elapsed,
            full_renders=fine['full_training_renders'],global_renders=fine['lod_training_renders'],
            local_renders=fine['view_budget_training_renders'],gi_refresh_renders=fine['importance_full_renders'],
            li_captures=fine['view_budget_li_full_renders'],li_vjps=fine['view_budget_li_backwards'],
            zero_global_steps=sum(int(v['K'])==0 for v in steps),zero_local_steps=sum(v['K']==0 for v in local),
            nonfinite=fine['lod_nonfinite'],global_K_total=sum(int(v['K']) for v in steps),local_K_total=sum(v['K'] for v in local),
            mean_mapping_ms=float(np.mean([float(v['ms']) for v in steps])),late_per_ratio=[])
        for ratio in TRAIN:
            chosen=[v for v in local[-2000:] if abs(v['ratio']-ratio)<1e-6]
            item['late_per_ratio'].append(dict(ratio=ratio,samples=len(chosen),K_mean=float(np.mean([r['K'] for r in chosen])),
                relative_budget_error=float(np.mean([abs(r['K']-r['target_K'])/max(1.,r['target_K']) for r in chosen])),
                photo_loss_mean=float(np.mean([r['photo_loss'] for r in chosen])),li_loss_mean=float(np.mean([r['li_loss'] for r in chosen]))))
        training.append(item)
        root=ROOT if group.startswith('raw_') else (PREVIOUS if group=='baseline' else old)
        manifest=root/'reports'/('SOURCE_HASHES.json' if group.startswith('raw_') else 'SOURCE_FINAL_HASHES.json')
        experiments.append(dict(group=group,seed=0,reused=item['reused'],input='physical11+view7' if group.startswith('raw_') else 'hidden32+view7' if group.startswith('view_') else 'no_view_branch',
            residual_parameters=321 if group.startswith('raw_') else 657 if group.startswith('view_') else 0,
            li_weight=.1 if group.endswith('li01') else 0,steps=11580,N=13298,anchors=2790,checkpoint=str(folder/'23560_shutdown/ply'),
            selector_config=str(folder/'23560_shutdown/ply/selector_config.yaml'),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            mapper_sha256=hashlib.sha256((root/'runtime/lib/libgaussian_mapper.so').read_bytes()).hexdigest(),
            full_renders=item['full_renders'],global_renders=item['global_renders'],local_renders=item['local_renders'],
            global_K_total=item['global_K_total'],local_K_total=item['local_K_total'],seconds=elapsed,status='complete'))
    (ROOT/'reports/TRAINING_AUDIT.json').write_text(json.dumps(training,indent=2))
    write('EXPERIMENTS.csv',experiments)
    metric=lambda g,c,k='psnr',part='all':index[(g,c,part)][k+'_mean']
    delta=lambda ref,g,c,k='psnr',part='all':paired_index[(ref,g,c,part)]['delta_'+k+'_mean']
    guards={g['group']:g for g in gates if g['reference']=='baseline'}
    matched=[g for g in gates if g['reference'] in ['view_no_li','view_li01']]
    conclusions=[]
    for g in matched:
        group,ref=g['group'],g['reference']
        both=g['view_1pct_delta']>0 and g['view_5pct_delta']>0
        verdict='1%和5%均提高' if both else '1%和5%未同时提高'
        conclusions.append(f'{NAMES[group]}相对{NAMES[ref]}：{verdict}，分别{g["view_1pct_delta"]:+.4f}/{g["view_5pct_delta"]:+.4f}dB；局部训练档平均{g["view_train_mean_delta"]:+.4f}dB。相对原联合baseline的全局质量筛查为{guards[group]["quality_guard_passed"]}。')
    lines=['# fr1_xyz：18D原始属性与39D中间特征视角残差对照','',
        '本轮未得到18D整体优于39D的证据。无LI时18D的低ratio PSNR更好，但LI=0.1时1%退化、5%基本持平；两个18D配置均未通过相对原联合基线的全局质量保护门槛。保留研究实现和模型，生产默认不切换。','',*conclusions,'',
        '本轮只新增两组18D完整训练；原联合baseline和两个39D模型及88704条逐帧指标只读复用。所有模型评价相同792帧，Bank72、non-Bank720；以下为本场景seed0单次结果，不能直接称为稳定收益。','',
        '## 主要配对结果','',
        '| 比较 | 局部训练六档ΔPSNR | 局部未见五档Δ | 局部1% Δ | 局部5% Δ | 全局六档Δ | 最差全局训练档Δ | Full Δ | 质量筛查 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for g in gates:
        lines.append(f'| {NAMES[g["group"]]} − {NAMES[g["reference"]]} | {g["view_train_mean_delta"]:+.4f} | {g["view_unseen_mean_delta"]:+.4f} | {g["view_1pct_delta"]:+.4f} | {g["view_5pct_delta"]:+.4f} | {g["global_train_mean_delta"]:+.4f} | {g["global_train_min_delta"]:+.4f} | {g["full_delta"]:+.4f} | {g["quality_guard_passed"]} |')
    lines+=['','单位dB。预注册正式保护门槛以原联合baseline为参照：全局六训练档均值≥−0.10dB、每个训练档≥−0.20dB、Full≥−0.10dB。其他参照的相同阈值列仅作诊断。低ratio和固定预算须分别判断，均值不能掩盖退化。','',
        '## 局部ratio质量','', '| ratio | 类型 | 原联合无LI | 39D无LI | 18D无LI | 39D LI=0.1 | 18D LI=0.1 | 18−39无LI | 18−39 LI=0.1 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    order=['baseline','view_no_li','raw_no_li','view_li01','raw_li01']
    for r in sorted(TRAIN+UNSEEN):
        case=f'view_{r:g}';v=[metric(g,case) for g in order]
        lines.append(f'| {r:g} | {"训练" if r in TRAIN else "未见"} | '+' | '.join(f'{q:.4f}' for q in v)+f' | {v[2]-v[1]:+.4f} | {v[4]-v[3]:+.4f} |')
    for scope,title in [('view','局部'),('global','全局')]:
        lines+=['',f'## {title}：匹配LI的SSIM / LPIPS变化','',
            '| ratio | 无LI ΔSSIM↑ | 无LI ΔLPIPS↓ | LI=0.1 ΔSSIM↑ | LI=0.1 ΔLPIPS↓ |','|---:|---:|---:|---:|---:|']
        for r in sorted(TRAIN+UNSEEN):
            case=f'{scope}_{r:g}';values=[delta(a,b,case,k) for a,b in [('view_no_li','raw_no_li'),('view_li01','raw_li01')] for k in ['ssim','lpips']]
            lines.append(f'| {r:g} | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines+=['','SSIM越高越好、LPIPS越低越好。全部绝对值、photo L1、Bank/non-Bank、median/p10/p90及改善帧比例见SUMMARY与PAIRED_SUMMARY。','',
        '## 全局与Full PSNR','', '| ratio | 原联合无LI | 39D无LI | 18D无LI | 39D LI=0.1 | 18D LI=0.1 |','|---:|---:|---:|---:|---:|---:|']
    for case,label in [(f'global_{r:g}',f'{r:g}') for r in sorted(TRAIN+UNSEEN)]+[('full','Full')]:
        lines.append('| '+label+' | '+' | '.join(f'{metric(g,case):.4f}' for g in order)+' |')
    lines+=['','## 局部固定点数PSNR','', '| 请求点数 | 原联合无LI | 39D无LI | 18D无LI | 39D LI=0.1 | 18D LI=0.1 | 无LI逐帧同K占比 | LI逐帧同K占比 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in [1000,2000,3000,4000,5000,10000]:
        case=f'points_{b}'
        values=[metric(g,case) for g in order]
        same=[paired_index[(a,g,case,'all')]['same_K_fraction'] for a,g in [('view_no_li','raw_no_li'),('view_li01','raw_li01')]]
        lines.append(f'| {b:,} | '+' | '.join(f'{v:.4f}' for v in values)+f' | {same[0]:.1%} | {same[1]:.1%} |')
    lines+=['','实际提交K=min(B,E)，10k可能候选不足；网络ratio沿用min(1,B/N)，保留超训练范围记录。不同训练后的地图和候选可不同，不能把相同请求预算视为相同选点集合或最终Alpha贡献数量。各自global贡献匹配预算的五档指标也保留在逐帧数据中，不称跨模型same-K。','',
        '## 同checkpoint开启−关闭视角残差','',
        '| 模型 | 局部配置 | ΔPSNR↑ | ΔSSIM↑ | ΔLPIPS↓ | 改善视角比例 |','|---|---|---:|---:|---:|---:|']
    for d in diagnostic:
        if d['partition']=='all':
            lines.append(f'| {NAMES[d["group"]]} | {d["case"]} | {d["delta_psnr_mean"]:+.4f} | {d["delta_ssim_mean"]:+.6f} | {d["delta_lpips_mean"]:+.6f} | {d["psnr_improved_fraction"]:.1%} |')
    lines+=['','这些诊断固定当前checkpoint、相机和K，只切换视角评分残差。它衡量当前残差在推理时的作用，不代表移除整个训练历史，也不能单独证明教师的边际价值。','',
        '## 真实训练预算','', '| 组 | 复用 | Full/全局/局部渲染 | LI采集/VJP | 全局累计K | 局部累计K | 零选点全局/局部 | 非有限值 | Mapping均值ms | 进程秒 |',
        '|---|---|---|---|---:|---:|---|---:|---:|---:|']
    for t in training:
        lines.append(f'| {NAMES[t["group"]]} | {t["reused"]} | {t["full_renders"]}/{t["global_renders"]}/{t["local_renders"]} | {t["li_captures"]}/{t["li_vjps"]} | {t["global_K_total"]} | {t["local_K_total"]} | {t["zero_global_steps"]}/{t["zero_local_steps"]} | {t["nonfinite"]} | {t["mean_mapping_ms"]:.3f} | {t["seconds"]:.2f} |')
    lines+=['','每组11580次Master/Selector/共享Adam更新、每ratio1930步；另有GI Full刷新864次。六Feature表独立Adam时钟11580/9650/7720/5790/3860/1930。Mapping步数相同不代表Gaussian曝光相同，累计K已单列。共享GPU运行时间不是配对性能Benchmark，不能据此断言训练加速。','',
        '## 实现与验收','',
        '- 新配置ViewBudget.view_residual_input=physical11；schema2对应18→16→1、321参数；默认hidden32/schema1保留39→16→1、657参数。原Selector11→32→32→2及Residual48补偿主体保留。',
        '- 11D为原Selector局部offset3、activated Scale3、SH DC3、activated Opacity1、log(ratio)1。C++直接复用当前pack结果，Python复用同一次forward的输入，避免额外的归一化或Gather。未加Anchor Residual48 Feature。',
        '- 相同7D Anchor相机编码：世界方向3、asinh归一化深度1、投影位置2、log1p投影大小1。固定场景尺度写入checkpoint，编码与物理输入detach，输出零初始化；全局评分不调用视角分支。',
        '- 局部Photo/Budget保留live全局logit与原Master/补偿梯度路径；LI只对残差四个参数Tensor定向VJP。Full/global/local反向之后一次共享Adam。',
        '- 原生CPU/CUDA测试覆盖零初始化选点身份、RNG、LI原有梯度逐bit不变、Photo/Budget路径、schema拒绝、保存恢复与冻结克隆；raw全量/compact/空输入打包核对，12步真实Mapping、36步真实checkpoint smoke及Python推理验收。详见VALIDATION_STATUS、INFERENCE_VALIDATION。',
        '- 两组18D原61个初始状态Tensor、Master SHA及11580步相机/ratio序列与baseline一致，新18D分支两组初态一致。39D新增分支参数形状不同，不能称与18D新增权重相同。fresh Adam/RNG，不称完整在线SLAM恢复。',
        '- 新增两个模型各39配置×792帧，61776条；旧88704条只读复用，总150480条。新增234次计数/普通Renderer颜色与radii逐bit检查；全部输入checkpoint和历史指标前后SHA核验，不保存渲染帧图片。','',
        '评价首次初始化失败：旧评价模块重排sys.path后，lazy import读取了旧版视角残差类。失败发生在新模型渲染前，只有88704条复用数据；现场保存在evaluation/failed_attempt_01及失败日志。评价入口改为提前导入并断言本轮view_residual绝对路径后恢复，仅重新执行评价，未重训或修改checkpoint。详见EVALUATION_REPAIR_01.json。','',
        '## 可确定结论与限制','',
        *conclusions,'',
        '- 18D输入减少21维、参数减少336个（约51.1%）；既改变输入信息，也改变容量，因此该实验检验的是完整18D设计，不能独立归因于“更少共享”或“物理输入更优”。',
        '- 物理属性仍随Master优化而变化；detach只隔离新增梯度路径，不能把输入看成静态状态。原始opacity、scale等也不能独立代表重建价值。',
        '- 对照复用历史二进制与运行，新旧源码/二进制SHA单独记录；通过旧路径回归不等于长训练逐bit可重复。单场景、单seed尚未隔离CUDA累加、共享GPU与历史运行波动。',
        '- 本轮质量评价显式使用PyTorch预算推理；原生Fine训练已验收，未重新验收18D在Viewer或冻结CUDA Graph加速路径上的部署性能。',
        '- Master、原Selector、Compensator及Feature仍共同训练，所以跨模型全部变化不能只归因于局部推理残差；同checkpoint开关诊断只隔离当前评分分支。',
        '- LI沿用Full alpha*T贡献排名，7D编码没有真实遮挡、邻点替代或图像误差信息。教师排序质量和点级网络可学习性仍需分别验证。',
        '- 未按non-Bank指标改学习率、权重、终点或挑checkpoint。生产默认未改，未commit/push，保护完整性详见FINAL_INTEGRITY.json。','',
        '## 结果归因','',
        '已证实：两组18D均未通过原联合baseline的全局质量保护；无LI时1%与5% PSNR较39D提高，但LPIPS分别上升约0.00450/0.00300，不能称为各项质量全面改善。LI=0.1时，18D的1% PSNR和SSIM较39D分别下降约0.3134dB/0.02696，LPIPS略有改善。两种LI设置的1%/5%逐帧实际K均与39D一致，差异不能用提交点数不同解释。','',
        '同checkpoint诊断进一步表明：无LI的18D残差在1%开启−关闭为+0.4411dB，5%为+0.2493dB；LI=0.1的18D残差在1%则为−0.1325dB，而39D＋LI为+0.1103dB。因此18D＋LI的1%问题包含当前视角评分分支本身的影响，不能只归咎于Master或补偿训练变化。另一方面，18D＋LI的1%残差开启后仍有58.6%的视角PSNR改善但均值下降，提示部分视角的大幅退化需要单独定位。','',
        '合理假设：更小的物理输入分支改变了LI标签与Photo/Budget共同优化时的可学习排序，低ratio的少量关键选择更敏感。尚未证实是输入信息、减少的容量、LI教师质量还是梯度冲突导致；本轮未做容量匹配对照或逐ratio均匀的冲突测量，也未把跨模型共享状态变化全部隔离。','',
        '## 下一步','',
        '优先固定Master、Compensator、原Selector、相机、候选和K，针对1%大幅退化视角比较LI教师集合与当前选点的实际图像质量，先判断教师是否有用，再检查18D分支能否学到有用排序。同时保留相同checkpoint残差开关，避免把共享地图变化混入分支归因。若确认教师有用而18D难以学习，再做容量匹配的输入对照；出现低ratio、固定预算和全局保护同时成立的候选后，再做配对重复验证。不要直接增大网络或同时改变多个Loss。','',
        f'模型目录：`{ROOT}/runs/full_raw_no_li/23560_shutdown/ply`、`{ROOT}/runs/full_raw_li01/23560_shutdown/ply`。','',
        '![ratio对照](figures/RAW18_RATIO_PSNR.png)','',
        '![固定点数](figures/RAW18_FIXED_BUDGET_PSNR.png)','',
        '![配对质量变化](figures/RAW18_METRIC_DELTAS.png)']
    (OUT/'RAW18_VIEW_RESIDUAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(matched,indent=2))


if __name__=='__main__':main()
