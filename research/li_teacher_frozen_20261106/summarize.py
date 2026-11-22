"""冻结教师诊断的逐视图配对、贡献与补偿归因，全部报告中文。"""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'evaluation/reports'
METRICS=['psnr','ssim','lpips','photo_l1','photo_train','K','projected_K','contributing_K',
    'teacher_overlap_fraction','full_li_selected_fraction','full_li_global_fraction','render_alpha_mass',
    'xyz_p99_reference','xyz_max_reference','scale_ratio_p99']
NAMES={'base':'原Selector／关闭残差','view':'18D视角残差','teacher':'Full LI教师'}


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(rows,key,out,prefix=None):
    a=np.array([float(r[key]) for r in rows]);assert np.isfinite(a).all()
    prefix=key if prefix is None else prefix
    for name,v in [('mean',a.mean()),('median',np.median(a)),('p10',np.quantile(a,.1)),('p90',np.quantile(a,.9)),('min',a.min()),('max',a.max())]:
        out[prefix+'_'+name]=float(v)


def partitions(rows):
    for name in ['all','bank','nonbank']:
        selected=[r for r in rows if name=='all' or (r['bank']=='True')==(name=='bank')]
        assert len(selected)==dict(all=792,bank=72,nonbank=720)[name]
        yield name,selected


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    with (OUT/'PER_VIEW.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==9504
    by=defaultdict(list);lookup={}
    for r in rows:
        key=(float(r['ratio']),r['selection'],r['compensation']=='True');by[key].append(r)
        lookup[(*key,int(r['frame']))]=r
    assert len(lookup)==9504 and len(by)==12
    summary=[]
    for (ratio,selector,comp),values in by.items():
        for partition,v in partitions(values):
            s=dict(ratio=ratio,selection=selector,compensation=comp,partition=partition,frames=len(v))
            for key in METRICS+['candidate_N','eligible_N','full_li_pool_global_fraction']:stats(v,key,s)
            s['inactive_teacher_frames']=sum(r['teacher_active']!='True' for r in v)
            s['training_float_budget_rounding_frames']=sum(r['training_float_ratio_teacher_K']!=r['teacher_K'] for r in v)
            summary.append(s)
    write('SUMMARY.csv',summary);(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    pairs=[]
    for ratio in [.01,.05]:
        for comp in [False,True]:
            for reference,selector in [('base','teacher'),('view','teacher'),('base','view')]:
                for frame in range(792):
                    a,b=lookup[(ratio,reference,comp,frame)],lookup[(ratio,selector,comp,frame)]
                    assert a['K']==b['K'] and a['candidate_N']==b['candidate_N'] and a['eligible_N']==b['eligible_N']
                    v=dict(ratio=ratio,reference=reference,selection=selector,compensation=comp,frame=frame,bank=a['bank'],K=a['K'])
                    for key in METRICS:v['delta_'+key]=float(b[key])-float(a[key])
                    pairs.append(v)
    write('PAIRED_PER_VIEW.csv',pairs)
    paired=defaultdict(list)
    for r in pairs:paired[(r['ratio'],r['reference'],r['selection'],r['compensation'])].append(r)
    paired_summary=[]
    for (ratio,reference,selector,comp),values in paired.items():
        for partition,v in partitions(values):
            s=dict(ratio=ratio,reference=reference,selection=selector,compensation=comp,partition=partition,frames=len(v))
            for key in METRICS:stats(v,'delta_'+key,s)
            s['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in v]));s['down_gt1dB']=sum(r['delta_psnr']<-1 for r in v)
            paired_summary.append(s)
    write('PAIRED_SUMMARY.csv',paired_summary);(OUT/'PAIRED_SUMMARY.json').write_text(json.dumps(paired_summary,indent=2))
    compensation=[]
    for ratio in [.01,.05]:
        for selector in NAMES:
            values=[]
            for frame in range(792):
                a,b=lookup[(ratio,selector,False,frame)],lookup[(ratio,selector,True,frame)]
                assert a['selection_sha256']==b['selection_sha256']
                v=dict(bank=a['bank'])
                for key in ['psnr','ssim','lpips','photo_train']:v['delta_'+key]=float(b[key])-float(a[key])
                values.append(v)
            for partition,v in partitions(values):
                s=dict(ratio=ratio,selection=selector,partition=partition,frames=len(v))
                for key in ['psnr','ssim','lpips','photo_train']:stats(v,'delta_'+key,s)
                s['psnr_improved_fraction']=float(np.mean([r['delta_psnr']>0 for r in v]));compensation.append(s)
    write('COMPENSATION_EFFECT.csv',compensation);(OUT/'COMPENSATION_EFFECT.json').write_text(json.dumps(compensation,indent=2))
    failed={i for i in range(792) if float(lookup[(.01,'view',True,i)]['psnr'])-float(lookup[(.01,'base',True,i)]['psnr'])<-1}
    failures=[]
    for i in sorted(failed):
        base=lookup[(.01,'base',True,i)];view=lookup[(.01,'view',True,i)];teacher=lookup[(.01,'teacher',True,i)]
        failures.append(dict(frame=i,bank=base['bank'],K=base['K'],base_psnr=base['psnr'],view_psnr=view['psnr'],teacher_psnr=teacher['psnr'],
            view_minus_base=float(view['psnr'])-float(base['psnr']),teacher_minus_base=float(teacher['psnr'])-float(base['psnr']),
            full_li_pool_global_fraction=teacher['full_li_pool_global_fraction']))
    if failures:write('FAILED_VIEWS.csv',failures)
    subgroup=[]
    for ratio in [.01,.05]:
        for comp in [False,True]:
            for ref in ['base','view']:
                for label,ids in [('known_1pct_failures',sorted(failed)),('remaining_views',sorted(set(range(792))-failed))]:
                    if not ids:continue
                    changes=[float(lookup[(ratio,'teacher',comp,i)]['psnr'])-float(lookup[(ratio,ref,comp,i)]['psnr']) for i in ids]
                    subgroup.append(dict(ratio=ratio,compensation=comp,reference=ref,subgroup=label,frames=len(ids),
                        delta_psnr_mean=float(np.mean(changes)),delta_psnr_median=float(np.median(changes)),improved_fraction=float(np.mean(np.array(changes)>0))))
    write('FAILURE_SUBGROUPS.csv',subgroup);(OUT/'FAILURE_SUBGROUPS.json').write_text(json.dumps(subgroup,indent=2))
    si={(r['ratio'],r['selection'],r['compensation'],r['partition']):r for r in summary}
    pi={(r['ratio'],r['reference'],r['selection'],r['compensation'],r['partition']):r for r in paired_summary}
    gates=[]
    for ratio in [.01,.05]:
        a=pi[(ratio,'base','teacher',True,'all')];b=pi[(ratio,'base','teacher',True,'nonbank')]
        gate=dict(ratio=ratio,mean_delta_psnr=a['delta_psnr_mean'],median_delta_psnr=a['delta_psnr_median'],
            improved_fraction=a['psnr_improved_fraction'],nonbank_delta_psnr=b['delta_psnr_mean'],
            delta_ssim=a['delta_ssim_mean'],delta_lpips=a['delta_lpips_mean'],delta_photo_train=a['delta_photo_train_mean'])
        gate['passed']=gate['mean_delta_psnr']>=.1 and gate['median_delta_psnr']>=0 and gate['improved_fraction']>.5 and gate['nonbank_delta_psnr']>0 and gate['delta_ssim']>=0 and gate['delta_lpips']<=0
        gates.append(gate)
    qualified=all(g['passed'] for g in gates)
    (OUT/'TEACHER_GATES.json').write_text(json.dumps(dict(qualified=qualified,ratios=gates),indent=2))
    metric=lambda r,s,c,k='psnr':si[(r,s,c,'all')][k+'_mean']
    lines=['# LI教师是否值得学习：固定18D＋LI模型诊断','',
        '结论：'+('本checkpoint在两档预算下均通过预声明教师质量筛查，值得继续验证学习方式。' if qualified else '本checkpoint未在两档预算下同时通过预声明筛查；不能直接把当前LI教师当成更优选点真值。'),'',
        '主模型固定为18D＋LI=0.1最终checkpoint，Master、原Selector、残差、补偿、Feature、Anchor、相机、候选池及seed0推理噪声全部固定。全轨迹792帧，Bank72/non-Bank720；每帧使用自己的名义Anchor候选N，K=min(floor(rN),eligibleN)。三种集合×补偿开关×1%/5%，共9504条新评价；零训练/Adam更新，不保存渲染帧图片。','',
        '## 直接渲染质量','', '| ratio | 补偿 | 原Selector PSNR | 视角残差 PSNR | LI教师 PSNR | 教师−原Selector | 教师−视角残差 |', '|---:|---|---:|---:|---:|---:|---:|']
    for r in [.01,.05]:
        for c in [False,True]:
            v=[metric(r,s,c) for s in ['base','view','teacher']]
            lines.append(f'| {r:g} | {"开" if c else "关"} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[2]-v[0]:+.4f} | {v[2]-v[1]:+.4f} |')
    lines+=['','## 主要比较：补偿开启时教师−原Selector','',
        '| ratio | ΔPSNR均值 | median | p10/p90 | 改善视角 | ΔSSIM↑ | ΔLPIPS↓ | Δ原生Photo Loss↓ | 筛查 |','|---:|---:|---:|---|---:|---:|---:|---:|---|']
    for g in gates:
        r=g['ratio'];p=pi[(r,'base','teacher',True,'all')]
        lines.append(f'| {r:g} | {g["mean_delta_psnr"]:+.4f} | {g["median_delta_psnr"]:+.4f} | {p["delta_psnr_p10"]:+.4f}/{p["delta_psnr_p90"]:+.4f} | {g["improved_fraction"]:.1%} | {g["delta_ssim"]:+.6f} | {g["delta_lpips"]:+.6f} | {g["delta_photo_train"]:+.6f} | {g["passed"]} |')
    lines+=['','筛查在渲染前声明：补偿开启时，两档teacher−base均值至少+0.10dB，median≥0、过半视角改善、non-Bank均值为正，且SSIM/LPIPS方向不矛盾。该门槛用于决定是否继续研究教师，不是生产非劣证明；每档和补偿状态必须单独解释。','',
        '## 补偿本身的效果','', '| ratio | 集合 | 开−关ΔPSNR | ΔSSIM↑ | ΔLPIPS↓ | ΔPhoto Loss↓ |','|---:|---|---:|---:|---:|---:|']
    for t in compensation:
        if t['partition']=='all':lines.append(f'| {t["ratio"]:g} | {NAMES[t["selection"]]} | {t["delta_psnr_mean"]:+.4f} | {t["delta_ssim_mean"]:+.6f} | {t["delta_lpips_mean"]:+.6f} | {t["delta_photo_train_mean"]:+.6f} |')
    lines+=['','## 贡献、实际点数与候选限制','',
        '| ratio | 集合 | 实际K均值 | 实际Alpha贡献点数（补偿开） | 保留候选Full贡献比例 | 候选覆盖整图Full贡献比例 | XYZ位移p99/参考尺度均值 | Scale倍率p99均值 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in [.01,.05]:
        for s in NAMES:
            a=si[(r,s,True,'all')]
            lines.append(f'| {r:g} | {NAMES[s]} | {a["K_mean"]:.2f} | {a["contributing_K_mean"]:.2f} | {a["full_li_selected_fraction_mean"]:.2%} | {a["full_li_pool_global_fraction_mean"]:.2%} | {a["xyz_p99_reference_mean"]:.4f} | {a["scale_ratio_p99_mean"]:.4f} |')
    lines+=['','教师保留的Full贡献质量较高是排序的直接结果，不等于图像质量提升。实际Alpha贡献点数也不能替代真实alpha×T。所有方法候选池、资格和K相同；本轮固定候选域，未隔离名义视锥遗漏的影响。Scale倍率只作诊断，不能不结合opacity和实际图像就认定为质量原因。','',
        '## 已知1%失败视角','',f'此前视角残差相对关闭残差下降超过1dB的{len(failed)}帧保留全部评价，未删除或用来改变教师、权重。失败帧明细见FAILED_VIEWS.csv；同一批帧在两档和补偿开关下的教师配对变化见FAILURE_SUBGROUPS.csv。','',
        '| ratio | 补偿 | 子集 | 教师比较参照 | 帧数 | ΔPSNR均值 | 改善比例 |','|---:|---|---|---|---:|---:|---:|']
    for s in subgroup:
        if s['compensation'] and s['subgroup']=='known_1pct_failures':
            lines.append(f'| {s["ratio"]:g} | 开 | 已知失败帧 | {NAMES[s["reference"]]} | {s["frames"]} | {s["delta_psnr_mean"]:+.4f} | {s["improved_fraction"]:.1%} |')
    status=json.loads((ROOT/'evaluation/STATUS.json').read_text());stability=json.loads((OUT/'TEACHER_STABILITY.json').read_text())
    changed=sum(not r['teacher_ids_equal'] for v in stability for r in v['ratios'])
    round_frames=max(v['training_float_budget_rounding_frames'] for v in summary)
    lines+=['','## 验收与局限','',
        f'- 原生排序7项单元测试；{status["native_teacher_sort_checks"]}次与Python稳定字典排序核对；{status["rasterizer_bitwise_checks"]}次Full/选点渲染颜色和radii逐bit核对。base/view补偿开时的3168条选点SHA和K与上一轮逐条一致；PSNR最大绝对差{status["prior_psnr_max_abs_error"]:.9g}dB（评价批次组织可不同，颜色逐bit检查单独执行）。',
        f'- 7个固定视角重复原生Full采集，14组ratio教师ID中{changed}组改变；归约浮点误差明细见TEACHER_STABILITY.json。若ID有变化，结论须结合边界不稳定性，不把原生atomic非确定性当作新Kernel数值差异。',
        f'- 正常推理沿用Python名义ratio计算floor(rN)；训练配置ratio保存为FP32，恰好整数边界时可能相差1点。本轮每档相关视角最多{round_frames}帧，已记录training_float_ratio_teacher_K；所有对照使用同一明确K。不是逐步Gumbel训练历史重放。',
        '- 主贡献采集使用未补偿Full的原生Training Forward、全像素有效Alpha及透射率；调用原localImportanceTarget，同分按保存的全局GI及固定source row，不新增深度或网络。已保存的GI来自训练最后一次刷新，只作平分排序，不重新估计。',
        '- LibTorch ABI1与Python Torch ABI0不能直接混用；首次PATH/Ninja和ABI失败日志均保留。最终将相同CUDA源码在独立build原样编为ABI0，未改Kernel，颜色/radii逐bit检查通过后才评价。生产及共享conda未改。',
        '- Photo Train是当前RGB评价输入上的原训练公式0.8 L1+0.2(1−原生SSIM)，图像未clamp；PSNR、TorchMetrics SSIM及LPIPS对clamp[0,1]图像计算。评价SSIM与训练SSIM不混用。',
        '- teacher直接读取当前视图Full贡献，有额外Full渲染，不能当成可部署免费推理，更不是全局最优点集。固定补偿没有专门为teacher集合重新训练；补偿失配也是本诊断要检测的因素。',
        '- 单checkpoint固定地图不证明真实在线收益、教师历史始终正确或学生一定学得会；本轮也未测局部分支Photo/Budget/LI梯度冲突。Bank与non-Bank均单独报告，不按non-Bank调参。',
        '- 源码、库、模型、既有CSV及604个生产文件前后SHA核验见FINAL_INTEGRITY.json；零Adam更新、不保存渲染帧图片，不commit/push。','',
        '## 本轮证据支持的结论','',
        '当前Full LI Top-K不适合直接作为既有选择＋补偿系统的更优选点真值。补偿开启后，1%/5%分别比原Selector低6.4652/2.9651dB，792帧全部退化；SSIM、LPIPS和原生Photo Loss方向也均更差。教师保留候选Full贡献质量32.90%/62.85%，明显高于原Selector的9.44%/43.37%，但没有转化成实际图像收益。','',
        '贡献信号并非完全没有信息。关闭补偿时，教师相对原Selector的PSNR分别−0.0641/−0.4330dB、LPIPS分别上升约0.03339/0.00369，但SSIM分别提高约0.19612/0.03219，原生Photo Loss分别下降约0.05311/0.00699。因此存在评价目标方向不一致；不能把结果概括为原始贡献在所有指标上都无效。','',
        '补偿与集合的适配缺口更明显：1%时原集合得到+8.9878dB补偿收益，教师集合只有+2.5867dB；5%分别为+6.5783/+4.0462dB。补偿本身仍提高教师质量，但远未达到原集合的收益。该结果支持继续检查贡献监督与当前补偿协同，不足以证明单独由Compensator导致，也不能推断针对teacher重新联合训练必然失败。','',
        '为排除新路径的实现错误，额外将教师ID强制送入原FixedBudgetRenderer：4视角×2ratio×补偿开关共16项，全属性Tensor、选点ID及渲染颜色均逐bit一致。四幅Full的Gaussian alpha质量和恒1颜色渲染Alpha均值最大绝对差约1.40e−5，两种计算采用不同归约顺序；该守恒检查差异单独记录，不用于接受新Kernel误差。结果见TEACHER_PAYLOAD_PARITY.json。','',
        '## 下一步判读','',
        '优先验证局部贡献与补偿后真实边际收益是否一致：固定当前地图和补偿，从原集合开始，以Full贡献作为候选提示，做有限同K点交换，只有实际Photo/PSNR改善才接受，先判断是否存在比原集合更好的局部点集；有限搜索不能称全局最优。当前证据不支持加大LI权重或直接强化对该Top-K的模仿。若贡献能提出有效交换，再设计弱集合约束或补偿感知的监督；原全局GI与网络主体保持对照。若后续得到更优教师，再进入学生可学习性和Loss对照。','',
        '![PSNR](figures/LI_TEACHER_PSNR.png)','', '![指标差异](figures/LI_TEACHER_DELTAS.png)']
    (OUT/'LI_TEACHER_VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(qualified=qualified,gates=gates,failed_frames=len(failed)),indent=2))


if __name__=='__main__':main()
