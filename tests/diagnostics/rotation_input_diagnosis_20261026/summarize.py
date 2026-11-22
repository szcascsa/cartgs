"""区分已测事实、表示对照与不能由交叉渲染推出的因果归因。"""
import csv
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from probe import ROOT,SOURCE,qrun,rt,representation,ids_for
from train import Network,config,PROTOCOL

KINDS=['baseline','rotation_all','rot6','cov6']
METRICS=['psnr','ssim','lpips','photo']
GROUPS=[('all',rt.RATIOS),('train',rt.TRAIN),('unseen',[r for r in rt.RATIOS if r not in rt.TRAIN])]


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def directory(kind,seed):
    if kind in ['baseline','rotation_all']:
        return SOURCE/'results/fr1_xyz'/qrun.PROTOCOL['configs'][kind]/str(seed)
    return ROOT/'results/fr1_xyz'/PROTOCOL['configs'][kind]/str(seed)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


@torch.no_grad()
def gauge_probe():
    scene=rt.Scene('fr1_xyz')
    q=scene.g.get_rotation.detach();turn=torch.zeros_like(q);turn[:,1]=1
    equivalent=rt._multiply_quaternion(q,turn)
    rows=[]
    for kind in KINDS:
        for seed in [3100,3101]:
            net=qrun.Network(scene,qrun.config(kind),seed) if kind in KINDS[:2] else Network(scene,config(kind),seed)
            net.load_state_dict(torch.load(directory(kind,seed)/'step_12000.pt',map_location='cuda')['state_dict'])
            net.eval()
            if kind=='rotation_all':
                saved=net.base_rotation
                alternative=qrun.previous.canonical_rotation(equivalent)
            elif kind in ['rot6','cov6']:
                saved=net.rotation_features
                alternative=representation(equivalent,scene.g.get_scaling.detach(),kind)
            for ratio in rt.RATIOS:
                ids0,p0,h0=ids_for(scene,net,ratio)
                delta0=net.compensate(h0,ids0,ratio)
                if kind=='rotation_all':net.base_rotation=alternative
                elif kind in ['rot6','cov6']:net.rotation_features=alternative
                ids1,p1,h1=ids_for(scene,net,ratio)
                delta_same=net.compensate(h1,ids0,ratio)
                delta1=net.compensate(h1,ids1,ratio)
                means=[]
                for v in scene.load_test():
                    a,_,_=scene.render(ids0,delta0,v);b,_,_=scene.render(ids1,delta1,v)
                    means.append(float(-10*torch.log10((b.clamp(0,1)-v['gt']).square().mean())+
                        10*torch.log10((a.clamp(0,1)-v['gt']).square().mean())))
                intersection=len(set(ids0.cpu().tolist())&set(ids1.cpu().tolist()))
                rows.append(dict(kind=kind,seed=seed,ratio=ratio,selected=len(ids0),
                    intersection=intersection,selected_recall=intersection/len(ids0),
                    probability_max_change=float((p1-p0).abs().max()),
                    compensation_on_original_rows_max_change=float((delta_same-delta0).abs().max()),
                    nonbank_psnr_change=float(np.mean(means))))
                if kind=='rotation_all':net.base_rotation=saved
                elif kind in ['rot6','cov6']:net.rotation_features=saved
            del net
            torch.cuda.empty_cache()
    assert scene.unchanged()
    rt.write_csv(ROOT/'analysis/equivalent_ellipsoid_gauge.csv',rows)
    return rows


def main():
    assert (ROOT/'train.exit').read_text().strip()=='0'
    assert json.loads((ROOT/'probe_complete.json').read_text())['master_frozen']
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    per_view,quality,groups,paired,fairness,curves=[],[],[],[],[],[]
    values={}
    for kind in KINDS:
        for seed in [3100,3101]:
            d=directory(kind,seed)
            info=json.loads((d/'complete.json').read_text())
            assert info['steps']==12000 and info['master_frozen']
            assert all(v==3000 for v in info['ratio_exposures'].values())
            assert all(v==(3000 if k.startswith('extra_tables.') else 12000) for k,v in info['adam_clocks'].items())
            fairness.append(dict(kind=kind,seed=seed,steps=info['steps'],parameters=info['parameters'],
                seconds=info['seconds'],exposure_per_ratio=3000,source='reused_previous' if kind in KINDS[:2] else 'new',
                adam_clocks=json.dumps(info['adam_clocks'])))
            if kind in KINDS[:2]:
                dest=ROOT/'references'/kind/str(seed);dest.mkdir(parents=True,exist_ok=True)
                for name in ['complete.json','step_12000.pt','bank_full.csv','validation_full.csv','test_full.csv',
                    'training_curve.csv','validation_per_view.csv']:
                    shutil.copy2(d/name,dest/name)
            curves.extend([dict(r,kind=kind,seed=seed) for r in read(d/'validation_per_view.csv')])
            for split in ['bank','validation','test']:
                rows=read(d/(split+'_full.csv'));values[kind,seed,split]=rows
                per_view.extend([dict(r,kind=kind,seed=seed) for r in rows])
                for policy in ['same_k','cached']:
                    for ratio in rt.RATIOS:
                        sub=[r for r in rows if r['policy']==policy and float(r['ratio'])==ratio]
                        counts={int(r['selected']) for r in sub};assert len(counts)==1
                        quality.append(dict(kind=kind,seed=seed,split=split,policy=policy,ratio=ratio,selected=counts.pop(),
                            **{m:np.mean([float(r[m]) for r in sub]) for m in METRICS}))
                    for group,ratios in GROUPS:
                        sub=[r for r in quality if r['kind']==kind and r['seed']==seed and r['split']==split
                            and r['policy']==policy and r['ratio'] in ratios]
                        groups.append(dict(kind=kind,seed=seed,split=split,policy=policy,group=group,
                            **{m:np.mean([r[m] for r in sub]) for m in METRICS}))
    comparisons=[('rotation_all','baseline'),('rot6','baseline'),('cov6','baseline'),
        ('rot6','rotation_all'),('cov6','rotation_all'),('cov6','rot6')]
    for split in ['bank','validation','test']:
        for ratio in rt.RATIOS:
            assert len({r['selected'] for r in quality if r['split']==split and r['policy']=='same_k' and r['ratio']==ratio})==1
        for a,b in comparisons:
            for policy in ['same_k','cached']:
                for group,ratios in GROUPS+[(str(r),[r]) for r in rt.RATIOS]:
                    for m in METRICS:
                        diffs=[]
                        for seed in [3100,3101]:
                            index=lambda k:{(r['path'],float(r['ratio'])):r for r in values[k,seed,split]
                                if r['policy']==policy and float(r['ratio']) in ratios}
                            x,y=index(a),index(b);assert x.keys()==y.keys()
                            diffs.append(np.array([float(x[k][m])-float(y[k][m]) for k in x]))
                        v=np.stack(diffs);sign=1 if m in ['psnr','ssim'] else -1
                        paired.append(dict(candidate=a,reference=b,split=split,policy=policy,group=group,metric=m,
                            mean=float(v.mean()),seed3100=float(v[0].mean()),seed3101=float(v[1].mean()),
                            median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)),
                            improved_fraction=float((sign*v>0).mean()),seed_improved=int((sign*v.mean(1)>0).sum())))
    assert all(np.isfinite(float(r[m])) for r in per_view for m in METRICS)
    cross=read(ROOT/'crossover_per_view.csv');cross_summary=[]
    for split in ['bank_train','test']:
        for group,ratios in GROUPS+[('1%',[.01])]:
            for selection in KINDS[:2]:
                for compensation in ['baseline','rotation_all','none']:
                    sub=[r for r in cross if r['split']==split and float(r['ratio']) in ratios
                        and r['selection']==selection and r['compensation']==compensation]
                    cross_summary.append(dict(split=split,group=group,selection=selection,compensation=compensation,
                        **{m:np.mean([float(r[m]) for r in sub]) for m in ['psnr','photo']}))
    gradients=read(ROOT/'gradient_snapshot.csv')
    # 零向量无方向，原始计算中的零占位不应被解释成正交。
    validated=[]
    for row in gradients:
        row=dict(row)
        for a,b in [('selection','compensation'),('budget','compensation'),('budget','selection')]:
            if float(row[a+'_grad_norm'])==0 or float(row[b+'_grad_norm'])==0:
                row[a+'_'+b+'_cosine']=''
        validated.append(row)
    gradient_summary=[]
    for kind in KINDS[:2]:
        for ratio in rt.TRAIN:
            sub=[r for r in validated if r['kind']==kind and float(r['ratio'])==ratio
                and r['parameter']=='encoder' and r['scope']=='all']
            keys=['selection_grad_norm','compensation_grad_norm','budget_grad_norm',
                'selection_compensation_cosine','budget_compensation_cosine']
            gradient_summary.append(dict(kind=kind,ratio=ratio,**{k:np.mean([float(r[k]) for r in sub]) for k in keys}))
    for name,rows in [('quality_per_view.csv',per_view),('quality_summary.csv',quality),('group_summary.csv',groups),
        ('paired_statistics.csv',paired),('fairness_checks.csv',fairness),('validation_curves.csv',curves),
        ('crossover_summary.csv',cross_summary),('gradient_snapshot_validated.csv',validated),('gradient_summary.csv',gradient_summary)]:
        rt.write_csv(out/name,rows)
    pick=lambda a,b='baseline',g='all':next(r for r in paired if r['candidate']==a and r['reference']==b
        and r['split']=='test' and r['policy']=='same_k' and r['group']==g and r['metric']=='psnr')
    summary=dict(primary=[pick(a,b,g) for a,b in comparisons for g,_ in GROUPS],
        source_frozen=True,same_k_equal=True,metrics_finite=True,new_models=4,reused_models=4,
        experiment='fr1_xyz frozen-map, two initialization/frame-order seeds, final12000',
        no_production_change=True,online_slam=False)
    rt.save_json(out/'summary.json',summary)
    gauge=gauge_probe()
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,seed in zip(axes,[3100,3101]):
        for kind in KINDS:
            sub=sorted([r for r in quality if r['kind']==kind and r['seed']==seed and r['split']=='test'
                and r['policy']=='same_k'],key=lambda r:r['ratio'])
            ax.plot([r['ratio']*100 for r in sub],[r['psnr'] for r in sub],marker='o',label=kind)
        ax.set(xlabel='Ratio (%)',ylabel='PSNR (dB)',title=f'fr1_xyz seed{seed}');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality_curves.png',dpi=150);plt.close(fig)
    stats=json.loads((ROOT/'input_statistics.json').read_text())
    lines=['# Rotation输入退化归因与编码对照','',
        '本轮不更改生产算法，先审计当前Quaternion输入，再进行两种预注册6D输入编码控制。所有质量结果为同源fr1_xyz冻结地图；历史评价轨迹反复用于结构研究，不能当作新盲测。','',
        '## 1. 当前已证实的事实','',
        '- Quaternion已单位归一化、按最大绝对分量为正统一q/-q；分量有界，标准差约0.29～0.35。退化不是未做单位化或明显数值长尾。',
        f'- 0.01rad随机小扰动的符号跳变占{stats["small_rotation_boundary_probe"][2]["antipodal_jump_fraction"]*100:.4f}%；SO3最近邻中转角<5°而Quaternion输入距离>1的点占{stats["nearest_rotation_neighbor"]["angle_lt5_and_distance_gt1_fraction"]*100:.4f}%。边界问题存在，但当前样本占比很小，不能判为主要原因。',
        f'- 最大/最小Scale轴比中位数{stats["anisotropy_quantiles"][1]:.2f}，轴比<1.2的近球形点仅{stats["nearly_isotropic_fraction_1p2"]*100:.2f}%；不能将Rotation视为大多数Gaussian上的无关信息。',
        '- 同一椭球的局部两轴反号后，Quaternion输入距离约sqrt(2)、Rot6距离2，而归一化协方差最大变化仅约6.6e-7；原Quaternion和Rot6会区分这种等价几何表示。','',
        '## 等价椭球输入的实际网络响应','',
        '保持真实Master与所有属性不变，仅将输入Rotation换成右乘局部X轴180°的等价表示；对Gaussian协方差没有实质改变。下表记录相同K下网络响应，属于反事实不变性诊断，不是训练收益。','',
        table(['方案','同K集合保留比例','最大概率变化','最大补偿分量变化','non-Bank ΔPSNR'],
            [[kind,f'{np.mean([r["selected_recall"] for r in gauge if r["kind"]==kind]):.4f}',
                f'{max(r["probability_max_change"] for r in gauge if r["kind"]==kind):.3g}',
                f'{max(r["compensation_on_original_rows_max_change"] for r in gauge if r["kind"]==kind):.3g}',
                f'{np.mean([r["nonbank_psnr_change"] for r in gauge if r["kind"]==kind]):+.3f}'] for kind in KINDS]),'',
        '该诊断能证明模型是否依赖任意的椭球轴符号，不能证明这种反事实变化在原训练轨迹中经常发生，也不能将反事实损失直接归为正常评价的PSNR差距。','',
        '## 2. 选点与补偿的交叉渲染','',
        table(['范围','选点来源','补偿来源','non-Bank PSNR'],[[r['group'],r['selection'],r['compensation'],f'{r["psnr"]:.4f}']
            for r in cross_summary if r['split']=='test' and r['group'] in ['all','1%']]),'',
        '- 1%无补偿时Rotation选点反而更好，但各自补偿后更差，不能只归因于选择了更差的基础Gaussian。两个模块相互交换后均明显退化，说明对各自训练选点支持集存在适配。',
        '- 交换模块会遇到另一分支较少训练的Gaussian/Anchor，属于分布外模块组合；不能将这些差值相加成纯选择/纯补偿因果贡献，也不是重新优化后的质量上界。','',
        '## 3. 共享梯度快照','',
        table(['方案','ratio','Selection梯度范数','Compensation梯度范数','Budget梯度范数','Selection/Comp cosine'],
            [[r['kind'],r['ratio'],f'{r["selection_grad_norm"]:.4f}',f'{r["compensation_grad_norm"]:.4f}',
                f'{r["budget_grad_norm"]:.4f}',f'{r["selection_compensation_cosine"]:.3f}'] for r in gradient_summary]),'',
        '- 原方案和Rotation方案均有负方向关系；Rotation方案的cosine没有更负，本快照不支持“加入Rotation让方向冲突更严重”的归因。',
        '- 1%补偿对Encoder梯度增大，但仅为最终checkpoint、每ratio固定4个Bank视图的快照；不能代替全训练过程，也不能仅凭范数/cosine认定更新有害。零梯度方向在validated CSV中留空。','',
        '## 4. 为什么不直接log Rotation','',
        '- ratio是正标量，log改变其尺度与变化分辨率。Quaternion包含负数且受单位约束，对每个分量取log不合法，也不保持旋转语义。',
        '- SO(3)旋转对数是另一种数学变换（轴角向量），不是逐分量log；全局角度接近π时仍有分支问题。本轮未将其作为默认修复。',
        '- 6D旋转采用R的前两列，只修正旋转坐标的连续性；归一化协方差同时考虑椭球Scale和朝向的实际几何。','',
        '## 5. 预注册输入编码对照','',
        '- Rot6：vec(R[:,0:2])，保持列顺序；Cov6：Sigma=R diag(scale²) Rᵀ，Sigma_bar=Sigma/trace(Sigma)，输入[xx,yy,zz,sqrt2 xy,sqrt2 xz,sqrt2 yz]。',
        '- 两组均原始11维+编码6维，Encoder17→32，四Head输入17+latent16+h32=65。新增输入列零初始化，其余公共权重与原实验同seed复制；14维输出、Loss、Adam、ratio顺序、插值不变。',
        '- 同维数Rot6/Cov6比较可检验表示语义；与Quaternion4比较同时多320参数，不能将全部增量归为纯连续性。相对无Rotation Baseline多960参数。',
        '- 新增4模型，各12000步、每ratio3000，seed3100/3101；复用上一轮相同协议的4个Baseline/Quaternion模型，保留全部结果，不选最好参考。','',
        table(['对照','seed3100 ΔPSNR','seed3101 ΔPSNR','平均ΔPSNR','训练ratio','未见ratio'],
            [[a+' − '+b,f'{pick(a,b)["seed3100"]:+.3f}',f'{pick(a,b)["seed3101"]:+.3f}',f'{pick(a,b)["mean"]:+.3f}',
                f'{pick(a,b,"train")["mean"]:+.3f}',f'{pick(a,b,"unseen")["mean"]:+.3f}'] for a,b in comparisons]),'',
        table(['方案','PSNR↑','SSIM↑','LPIPS↓','photo↓'],[[kind,*[f'{np.mean([r[m] for r in groups if r["kind"]==kind and r["split"]=="test" and r["policy"]=="same_k" and r["group"]=="all"]):.5f}'
            for m in METRICS]] for kind in KINDS]),'',
        '## 6. 证据边界','',
        '- Master、拓扑、Anchor关联、相机、cached uniform冻结。Selector/Encoder仍训练，集合可变；same-K按logit(p)-logit(cached uniform)排序，只保证预算一致。原Cached实际点数另存。',
        '- 相同54 Bank训练、13内部留出、45历史non-Bank。Bank字段为67视图（包含13留出）；最终12000步，未按non-Bank选LR、步数或checkpoint。',
        '- 高斯椭球局部轴符号对称性被Cov6输入消除；原始Scale向量仍保留，不能宣称整个网络已对所有轴置换不变。',
        '- 两个seed共享一张冻结地图，不能当独立SLAM重复或跨场景稳定结论。GPU有其他任务，耗时不作无干扰性能结论。',
        '- 输入几何问题、优化状态与选点支持集共同适配可以并存；负cosine、交叉渲染损失或表示连续性本身均不能单独证明主因。','',
        '## 参考依据','',
        '- [Zhou等：旋转表示的连续性](https://arxiv.org/abs/1812.07035)：支持测试6D坐标的数学动机，不证明本项目PSNR必然提高。',
        '- [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)：Gaussian形状由协方差描述；本项目实际约定另由CUDA源码和预计算协方差渲染gate核对。','',
        '![质量曲线](analysis/quality_curves.png)','']
    (ROOT/'ROTATION_INPUT_DIAGNOSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    with tarfile.open(ROOT/'evidence.tar.gz','w:gz') as pack:
        for path in ROOT.rglob('*'):
            if path.is_file() and (path.suffix in ['.py','.json','.csv','.md','.png','.npz'] or path.name=='step_12000.pt'):
                pack.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    print('PRIMARY',json.dumps(summary['primary']),flush=True)
    print('ARCHIVE_SHA256',rt.sha(ROOT/'evidence.tar.gz'),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'summarize.exit').write_text('1');raise
    (ROOT/'summarize.exit').write_text('0')
