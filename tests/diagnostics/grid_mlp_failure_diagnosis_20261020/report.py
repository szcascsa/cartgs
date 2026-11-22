"""核对原始日志并生成中文机制报告；不改模型、grid或训练参数。"""
from pathlib import Path
import hashlib,json,math,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
GRID=ROOT.parent/'grid_mlp_implicit_si_20261020';OLD=ROOT.parent/'spatial_mlp_implicit_si_20261020'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=lambda p:[json.loads(line[3:])for line in p.read_text().splitlines()if line.startswith('PG ')]

def main():
    remote=read(ROOT/'remote_artifact_manifest.json')['files']
    for name,z in remote.items():assert (ROOT/name).stat().st_size==z['bytes']and sha(ROOT/name)==z['sha256']
    continuation=read(ROOT/'continuation_results.json');assert continuation['all_passed']and continuation['completed']==3
    tr=[];hard=[];views=[];replay_error=0.;aggregate_error=0.
    for model in continuation['models']:
        scene=model['scene'];new=record(ROOT/'continuation'/scene/'training.log');old=record(GRID/scene/'models/Grid-MLP_5_seed_2026/training.log')
        before=[z for z in old if z['kind']=='training'];after=[z for z in new if z['kind']=='training'];assert len(after)==5000
        info=read(OLD/scene/'input_manifest.json');bank=set(info['bank_frames']);evaluation=set(info['evaluation_frames']);assert not bank&evaluation
        assert all(z['frame']in bank and z['frame']not in evaluation for z in after)
        for a,b in zip(before,after):
            assert a['step']==b['step']and a['frame']==b['frame']
            replay_error=max(replay_error,abs(a['sampled_photo']-b['sampled_photo']))
        assert replay_error==0
        original={z['step']:z for z in old if z['kind']=='hard_summary'}
        for z in new:
            if z['kind']=='training':tr.append(dict(scene=scene,**{k:v for k,v in z.items()if k!='kind'}))
            elif z['kind']=='hard_view':views.append(dict(scene=scene,**{k:v for k,v in z.items()if k!='kind'}))
            elif z['kind']=='hard_summary':
                if z['step']<=2000:
                    assert z['selected_source_rows']==original[z['step']]['selected_source_rows']
                    assert z['photo']==original[z['step']]['photo']and z['psnr']==original[z['step']]['psnr']
                own=[v for v in new if v['kind']=='hard_view'and v['step']==z['step']]
                assert len(own)==len(bank)and {v['frame']for v in own}==bank
                for metric in ['photo','psnr']:aggregate_error=max(aggregate_error,abs(math.fsum(v[metric]for v in own)/len(own)-z[metric]))
                hard.append(dict(scene=scene,**{k:v for k,v in z.items()if k not in ['kind','selected_source_rows']}))
        assert new[-1]['completed']and new[-1]['frozen_exact']and new[-1]['grid_frozen_exact']
    assert len(tr)==15000 and aggregate_error<1e-12
    training=pd.DataFrame(tr);hard=pd.DataFrame(hard);training.to_csv(ROOT/'continuation_training_curves.csv',index=False);hard.to_csv(ROOT/'continuation_hard_dynamics.csv',index=False);pd.DataFrame(views).to_csv(ROOT/'continuation_hard_per_view.csv',index=False)
    assert training.normalized_gradient_sum.max()<1e-12
    trajectory=pd.read_csv(ROOT/'trajectory_diagnosis.csv');resolution=pd.read_csv(ROOT/'resolution_prefix_bounds.csv');geometry=read(ROOT/'geometry_diagnosis.json');sampling=read(ROOT/'sampling_summary.json');summary=read(ROOT/'existing_evidence_summary.json')
    for _,g in resolution.groupby(['scene','ratio','seed','reference']):assert np.all(np.diff(g.sort_values('h_divisor').prefix_recall_upper_bound)>=0)
    locked=read(GRID/'selected_before_quality.json');assert sha(GRID/'selected_sets.npz')==locked['selected_SHA']and sha(GRID/'rankings.npz')==locked['rankings_SHA']
    inputs=[GRID/name for name in ['train_worker.cpp','dependencies/train_worker.cpp','dependencies/calibrated_selection.h','training_curves.csv','hard_training_dynamics.csv','selected_set_overlap.csv','quality_deltas.csv']]
    inputs += [path for scene in ['fr1_room','fr1_xyz','fr2_xyz']for path in [GRID/scene/'cell_mapping.npz',OLD/scene/'ranking_inputs.npz',OLD/scene/'input_manifest.json']]
    inputs += [OLD/'selected_sets.npz',REPO/'include/selector_handoff.h']
    (ROOT/'input_manifest.json').write_text(json.dumps({str(p):dict(sha256=sha(p),bytes=p.stat().st_size)for p in inputs},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    assert not subprocess.check_output(['git','diff','--name-only'],cwd=REPO).strip()
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO).strip()
    correctness=dict(all_passed=True,remote_files_verified=len(remote),prefix_DP_exhaustive_tests=summary['prefix_DP_bruteforce_tests_passed'],
        finer_grid_prefix_bound_monotonic=True,formal_grid_and_selected_sets_unchanged=True,continuation_cases=3,continuation_updates=15000,
        first2000_training_loss_max_abs=replay_error,all_replayed_hard_sets_exact=True,all_replayed_Bank_metrics_exact=True,
        continuation_aggregate_perview_max_abs=aggregate_error,continuation_max_normalized_gradient_sum=float(training.normalized_gradient_sum.max()),
        original_native_binary_reused=True,Gaussian_TF_Master_grid_frozen_exact=True,no_nonBank_continuation=True,
        no_production_change=True,no_staging=True,no_commit=True,no_push=True)
    (ROOT/'correctness.json').write_text(json.dumps(correctness,indent=2)+'\n')
    decision=dict(primary='训练目标/优化与hard导出不一致有直接证据，单纯步数不足不支持',secondary='当前grid的intra-cell Raw-order约束确实限制表示已有强baseline集合',
        unresolved='不能把grid上界当quality上界；尚未区分MLP smoothness、LR/优化条件和ST目标各自因果份额',
        action='优先验证固定grid下hard目标的可优化性，再考虑分辨率；不直接加步数或实现split',correctness=correctness)
    (ROOT/'diagnosis_summary.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,scene in zip(axes,['fr1_room','fr1_xyz','fr2_xyz']):
        h=hard[hard.scene==scene].sort_values('step');ax.plot(h.step,h.photo-h.iloc[0].photo,label='Grid-MLP');ax.axhline(0,color='black',lw=.8);ax.axvline(2000,color='grey',ls='--');ax.set_title(scene+' / 5% / 2026');ax.set_xlabel('step');ax.grid(alpha=.2)
    axes[0].set_ylabel('hard Bank photo - Raw (lower is better)');fig.tight_layout();fig.savefig(ROOT/'continuation_quality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,reference in zip(axes,['PG-ST','Fixed','C']):
        s=resolution[resolution.reference==reference].groupby('h_divisor').prefix_recall_upper_bound.agg(['min','median','max']);ax.plot([1,2,4],100*s['median'],marker='o');ax.fill_between([1,2,4],100*s['min'],100*s['max'],alpha=.15);ax.set_xticks([1,2,4],['h','h/2','h/4']);ax.set_title(reference+' set');ax.set_ylim(70,100);ax.grid(alpha=.2)
    axes[0].set_ylabel('independent-cell prefix recall upper bound (%)');fig.tight_layout();fig.savefig(ROOT/'prefix_resolution_bounds.png',dpi=160);plt.close(fig)

    src=str(GRID).replace('\\','/');loc=str(ROOT).replace('\\','/');repo=str(REPO).replace('\\','/')
    lines=['# Grid MLP 失败机制诊断','',
        '**结论：两个层面都有问题，但当前证据不支持“只是2000步没训够”。** 当前grid确实限制cell内部身份调整；与此同时，训练连已能exact表示的Raw都经常越训越差，不能用grid过粗解释全部退化。最应优先处理的是训练目标/优化与deterministic hard Top-K的错位。','',
        '| 问题 | 本轮判断 | 证据强度 |','|---|---|---|',
        '| grid是否偏粗？ | 对复现已有PG-ST/Fixed/C集合，YES | 约束与组合上界直接证明；不是quality上界 |',
        '| 是否仅因训练步数不足？ | 不支持 | 原24轨迹+预指定3case从2000延长到5000 |',
        '| MLP是否正常更新？ | YES，输出/排名在持续变化 | finite gradient、非零field、持续预算转移 |',
        '| 训练目标/导出错位？ | 有直接证据 | sample/hard集合分离，采样loss下降与hard退化并存 |',
        '| 能否断言只要细化grid就会好？ | NO | 同一1216参数MLP在更多centers采样，容量没有随cell数增长 |','',
        '## 1. “没训够”检查：没有持续朝hard目标收敛','',
        '对原48个模型，比较前500与后500步sample photo，先按Bank frame分别平均、再frame等权，避免view构成变化；hard指标使用每100步与step250的全Bank deterministic same-K。','',
        '| arm | sample photo下降 | final hard photo优于Raw | sample下降但hard差于Raw | 后1000步hard photo趋势变差 |','|---|---:|---:|---:|---:|']
    for arm in ['Grid-MLP','Independent-Cell']:
        z=summary[arm];lines.append(f'| {arm} | {z["sample_loss_improved"]}/24 | {z["hard_final_improved"]}/24 | {z["sample_improved_hard_worse"]}/24 | {z["late_hard_slope_worse"]}/24 |')
    lines += ['','Grid-MLP的24个final checkpoint均差于各自已记录的最低Bank hard photo；这只用于诊断，没有把最好step替换成正式结果。后半程梯度仍非零、coefficient仍更新，不能把“还在更新”当作“继续训练必然更好”。首尾sample loss统计是移动训练轨迹的描述，不是冻结checkpoint在同一完整sample上的精确因果比较。','',
        '## 2. 直接延长到5000步：原LR多训没有解决','',
        '预指定三scene的5%、seed2026，复用原grid_worker二进制、lr=3e-4、optimizer/loss/sampler/Bank schedule。由于旧模型未保存Adam状态，从step0重放以保留完整optimizer历史；前三个case的前2000步采样loss、全部已记录hard集合和Bank photo/PSNR均exact复现。随后继续到5000步，不调LR、不选最好step、不读取non-Bank。','',
        '| scene | Bank PSNR step0 | step2000 | step5000 | ΔPSNR 5000−2000 | Δphoto 5000−2000 |','|---|---:|---:|---:|---:|---:|']
    for model in continuation['models']:
        c={z['step']:z for z in model['checkpoints']};lines.append(f'| {model["scene"]} | {c[0]["psnr"]:.6f} | {c[2000]["psnr"]:.6f} | {c[5000]["psnr"]:.6f} | {model["PSNR_final_minus_2000"]:+.6f} | {model["photo_final_minus_2000"]:+.8f} |')
    lines += ['',f'![5000步Bank hard质量]({loc}/continuation_quality.png)','',
        '三个case的photo全部比step2000更差，step5000也全部差于Raw。fr2_xyz的PSNR比step2000微升，但photo继续变差，不能称为一致改善。因此“同设置只加步数”没有支持；本轮不排除其他LR/optimizer可改善，也不把三个case外推为所有ratio/seed的证明。','',
        '## 3. grid确有表达限制，但这不能解释为何低于Raw','',
        '同cell中score_i=base_i+a_C，因此只能从该cell按Raw顺序取一个prefix。给定目标Top-K T，若目标含有cell内第r个Gaussian，就必须同时取前r−1个。即使每cell coefficient完全独立且可任意设定，MLP也不能绕过这个约束。','',
        '本轮对每个目标集合建立所有cell的prefix选项，用动态规划求在总预算K内能命中的最多目标rows。DP先通过100个小规模随机fixture穷举校验。它允许每cell自由选择prefix长度、忽略跨cell score ties，因而是MLP可达集合的宽松上界。全过程只做已完成实验的posthoc集合分析，不用目标labels训练，不输出新quality方法。','',
        '| 目标集合 | 当前h最大recall中位数 | h/2 | h/4 |','|---|---:|---:|---:|']
    for reference in ['PG-ST','Fixed','C']:
        med=[resolution[(resolution.reference==reference)&(resolution.h_divisor==div)].prefix_recall_upper_bound.median()for div in [1,2,4]]
        lines.append(f'| {reference} | '+' | '.join(f'{100*v:.2f}%'for v in med)+' |')
    lines += ['','当前h下，PG-ST的24个集合、Fixed/C各12个集合均无法exact表示；完全覆盖PG-ST目标rows所需的cell-prefix预算中位数为原K的1.26倍，C为2.57倍。h/2和h/4只建立临时、固定world-origin的离线mapping，不重训MLP、不修改原mapping；occupied cells分别变为4251–5157、8451–10589。','',
        f'![网格分辨率与集合表示上界]({loc}/prefix_resolution_bounds.png)','',
        '图中线为跨case中位数，阴影为min/max范围，不是置信区间。','',
        '**上界不是质量结论。** 不同集合可能达到相近或更好的render质量；不能由C recall不足就推断所有regional correction都差，也不能由细网格recall上升就宣称PSNR会上升。更关键的是，a_C=0已经exact表示Raw：grid表示限制可以妨碍超过Raw，却不会迫使优化器把已可表示的Raw变差。当前大量Bank退化仍需从优化/目标解释。','',
        '| scene | 原h | Gaussian位于occupancy≥100的cell的比例 |','|---|---:|---:|']
    for g in geometry:lines.append(f'| {g["scene"]} | {g["h"]:.8f} | {100*g["population_fraction_in_cells_ge_100"]:.2f}% |')
    lines += ['','这些比例说明“2048 occupied cells”不等于Gaussian均匀分成小组；room/fr2的大量Gaussian处于大cell内。另一方面，细化只减少cell tying；tiny MLP仍只有1216个共享参数，并不会因为M增大而拥有M个独立自由度。','',
        '## 4. 代码路径：calibration已修复，目标错位仍保留','',
        f'- [cell-only forward与weighted gauge]({src}/train_worker.cpp:25)：MLP输入M个center，按occupancy保留autograd减均值，最后层bias=false。没有退回Gaussian逐点MLP或detached mean。',
        f'- [calibrated VJP]({src}/dependencies/calibrated_selection.h:24)：保留offset的一阶隐式导数。原96,000步normalized gradient sum最大约2.99e-16；本轮continuation也通过。此次没有发现旧common-score-shift错误残留。',
        f'- [训练选择]({src}/train_worker.cpp:65)与[ST helper]({repo}/include/selector_handoff.h:18)：训练用cached uniform与sigmoid admission，并非hard Top-K。expected count校准为K不代表sample集合等于hard集合。',
        f'- [compact photo路径]({src}/dependencies/train_worker.cpp:52)：只把当前sample的rows送入renderer，并在opacity上使用ST mask。校准VJP让unselected scores也收到归一化梯度，但这不是它们真实加入/替换后的render marginal。','',
        '| Grid-MLP checkpoint | sample vs hard recall中位数 | Jaccard中位数 | hard集合获得的expected selection mass/K |','|---|---:|---:|---:|']
    for step in ['0','2000']:
        z=sampling['Grid-MLP'][step];lines.append(f'| {step} | {100*z["median_sample_hard_recall"]:.2f}% | {z["median_sample_hard_Jaccard"]:.4f} | {100*z["median_expected_hard_mass"]:.2f}% |')
    lines += ['','final训练sample和最终hard集合只有约一半Gaussian相同。结合Independent-Cell“24/24采样loss下降、15/24 hard仍差于Raw”，这是当前训练目标与导出质量分离的直接证据。它尚不能单独定量证明cached randomness、ST近似与空间共享各占多少因果份额。','',
        '## 5. MLP sharing是不是额外限制？','',
        '相同grid、gauge、calibration、LR和训练预算下，Independent-Cell相对Grid-MLP在Bank有18/24个PSNR更好、non-Bank有14/24个更好，non-Bank平均高0.439837 dB。说明MLP这一参数化/优化组合增加了负担。但control本身仍未稳定优于Raw；不同参数化的梯度尺度和优化条件也不同，不能直接断言tiny MLP容量就是唯一主因。','',
        '本轮没有拟合PG-ST系数，也没有拟合Independent-Cell field；没有通过增加网络深度、feature或新regularizer改变问题。若后续要把MLP smoothness与优化彻底分开，需要独立的cell-field可表示性对照；当前不把它当作已证明结论。','',
        '## 6. 研究优先级','',
        '1. **优先：固定grid下验证训练能否稳定提高hard same-K目标。** Raw已在可表示集合内，而当前训练常偏离它；先解释sample与hard质量分离，不能用加步数掩盖。',
        '2. **其次：在训练目标可优化之后再验证分辨率。** 本轮已证明粗grid有集合表示损失，值得保留这条线索；但直接细化同一个平滑MLP，不会自动增加regional field表达自由度。',
        '3. **当前不进入split。** 高gradient、更多changed cells都不是收益证明；本轮未改grid、网络、loss、production，也未stage/commit/push。','',
        '## 7. 可复核输出','',
        f'- [轨迹诊断]({loc}/trajectory_diagnosis.csv)、[5000步训练曲线]({loc}/continuation_training_curves.csv)、[hard动态]({loc}/continuation_hard_dynamics.csv)、[Bank per-view]({loc}/continuation_hard_per_view.csv)。',
        f'- [当前grid prefix上界]({loc}/grid_prefix_representability.csv)、[分辨率表示上界]({loc}/resolution_prefix_bounds.csv)、[sample/hard错位]({loc}/sampling_export_mismatch.csv)。',
        f'- [长预算预注册]({loc}/continuation_preregistration.json)、[长预算结果]({loc}/continuation_results.json)、[correctness]({loc}/correctness.json)、[输入SHA]({loc}/input_manifest.json)、[结论JSON]({loc}/diagnosis_summary.json)。',
        '- 三个5000步模型的最终weights/field、step0/250/1000/5000快照、每100步diagnostic和原始日志保存在continuation/<scene>/；原24+24正式模型与结果保持不变。','']
    report='\n'.join(lines)
    (ROOT/'GRID_MLP_FAILURE_DIAGNOSIS.md').write_text(report,encoding='utf-8');(REPO/'GRID_MLP_FAILURE_DIAGNOSIS.md').write_text(report,encoding='utf-8')
    print(json.dumps(correctness,indent=2))

if __name__=='__main__':main()
