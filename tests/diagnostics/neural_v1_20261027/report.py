"""根据实际证据生成中文报告；不选择新结构或最佳checkpoint。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
TRAIN=[.01,.05,.1,.15]


def markdown(rows,cols):
    out=['| '+' | '.join(label for key,label in cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in rows:
        vals=[]
        for key,_ in cols:
            value=row.get(key,'');vals.append(f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value))
        out.append('| '+' | '.join(vals)+' |')
    return '\n'.join(out)


def main():
    state=json.loads((ROOT/'research_state.json').read_text());elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state['started_utc'])).total_seconds()/60
    q=pd.read_csv(ROOT/'quality_macro.csv');q=q[(q.policy=='same_k')&(q.split=='test')]
    ledger=pd.read_csv(ROOT/'NEURAL_V1_EXPERIMENTS.csv');completed=ledger[ledger.status.isin(['FORMAL_EVALUATED','TRAINED_BANK_ONLY'])]
    def model(tier,name,scene='fr1_xyz'):
        x=q[(q.tier==tier)&(q.config==name)&(q.scene==scene)];assert not x.empty,(tier,name,scene)
        base=dict(tier=tier,scene=scene,name=name,seeds=int(x.seed.nunique()))
        for g,key in [('seven','seven'),('train','train'),('unseen','unseen'),('1pct','p1'),('5pct','p5')]:
            v=x[x.group==g];base[key]=v.psnr.mean()
            if g=='seven':base.update(ssim=v.ssim.mean(),lpips=v.lpips.mean(),photo=v.photo.mean())
        return base
    def reference(scene):
        seed=3100 if scene=='fr1_xyz' else 4100
        f=pd.read_csv(ROOT/'references'/scene/'R3'/str(seed)/'test_per_view.csv');f=f[f.policy=='same_k']
        a=f.groupby('ratio')[['psnr','ssim','lpips','photo']].mean();return dict(scene=scene,name=f'R3 snapshot {seed}',seeds='1 source',
            seven=a.psnr.mean(),train=a.loc[TRAIN].psnr.mean(),unseen=a.loc[[.025,.075,.125]].psnr.mean(),
            p1=a.loc[.01].psnr,p5=a.loc[.05].psnr,ssim=a.ssim.mean(),lpips=a.lpips.mean(),photo=a.photo.mean())
    rows=[];lookup={}
    for scene,tier,ref_tier,seeds in [('fr1_xyz','confirm','reference_control',[5100,5101]),('fr2_xyz','fr2_confirm','fr2_confirm',[6100,6101])]:
        values=[model(ref_tier,'Reference59TrainEncoder',scene),model(tier,'SpatialFiLM',scene),model(tier,'SpatialOutputKnots',scene),
            model('controls' if scene=='fr1_xyz' else 'fr2_local_followup','SpatialLocalXYZ',scene),reference(scene)]
        rows.extend(values)
        for value in values:lookup[scene,value['name']]=value
    pd.DataFrame(rows).to_csv(ROOT/'main_results.csv',index=False)
    longrows=[model('long36k',name) for name in ['Reference59TrainEncoderLong','SpatialFiLMLong','SpatialOutputKnotsLong']]
    controls=[]
    for tier,name,base_tier,base_name in [
        ('confirm','PhysicalFiLM','confirm','RawPure'),('confirm','SpatialFiLM','confirm','PhysicalFiLM'),
        ('confirm','CrossAttentionFiLM','confirm','PhysicalFiLM'),('controls','SpatialNoFiLM','confirm','SpatialFiLM'),
        ('controls','SpatialNoFiLMCapacity','confirm','SpatialFiLM'),('controls','SpatialCapacityControl','confirm','SpatialFiLM'),
        ('controls','SpatialLocalXYZ','confirm','SpatialFiLM'),('controls','SpatialKnotsBands10','confirm','SpatialOutputKnots'),
        ('node_clock','SpatialKnotsSparseClock','confirm','SpatialOutputKnots'),('folded','SpatialKnotsFolded','confirm','SpatialOutputKnots'),
        ('input_cause','NoWorldControl','confirm','SpatialFiLM'),('input_cause','NoShapeControl','confirm','SpatialFiLM'),
        ('input_cause','NormalizedLocalOnly','confirm','SpatialFiLM')]:
        new=model(tier,name);old=model(base_tier,base_name)
        controls.append(dict(name=name,base=base_name,delta=new['seven']-old['seven'],train_delta=new['train']-old['train'],unseen_delta=new['unseen']-old['unseen']))
    pd.DataFrame(controls).to_csv(ROOT/'mechanism_comparisons.csv',index=False)
    baserows=[model('baseline_sets',name) for name in ['Reference59TrainEncoder','SpatialFiLM','SpatialOutputKnots']]
    repeats=[]
    for name,tier in [('Reference59TrainEncoder','reference_control'),('SpatialFiLM','confirm'),('SpatialOutputKnots','confirm')]:
        a=q[(q.tier==tier)&(q.config==name)&(q.seed==5100)&(q.group=='seven')].psnr.iloc[0]
        b=q[(q.tier=='repeat')&(q.config==name)&(q.seed==5100)&(q.group=='seven')].psnr.iloc[0]
        repeats.append(dict(name=name,first=a,repeat=b,delta=b-a))
    timing=pd.read_csv(ROOT/'mechanism/timing.csv');mem=pd.read_csv(ROOT/'mechanism/memory.csv');cost=[]
    for _,row in mem.iterrows():
        a=timing[(timing.config==row.config)&(timing.stage=='forward')];b=timing[(timing.config==row.config)&(timing.stage=='forward_backward_photo_vjp')]
        cost.append(dict(name=row.config,parameters=int(row.parameters),params_mib=row.parameter_bytes/2**20,adam_mib=row.adam_moments_bytes/2**20,
            forward_ms=a.wall_mean_ms.mean(),backward_ms=b.wall_mean_ms.mean()))
    topology=json.loads((ROOT/'final_validation.json').read_text())['topology']
    f1=lookup['fr1_xyz','SpatialFiLM'];f2=lookup['fr2_xyz','SpatialFiLM'];base1=lookup['fr1_xyz','Reference59TrainEncoder'];base2=lookup['fr2_xyz','Reference59TrainEncoder']
    local1=lookup['fr1_xyz','SpatialLocalXYZ'];local2=lookup['fr2_xyz','SpatialLocalXYZ'];r31=reference('fr1_xyz');r32=reference('fr2_xyz')
    maximum_repeat=max(abs(x['delta']) for x in repeats)
    endpoint=json.loads((ROOT/'endpoint_support_probe_summary.json').read_text())
    endpoint_rows=[dict(source=v['source'],ratio=v['ratio'],normal=v['normal'],
        unsupported=v['replace_lower_unsupported'],supported=v['replace_lower_supported'],
        delta=v['replace_lower_unsupported']-v['normal']) for v in endpoint]
    text=f'''# Neural Compensation V1：发现驱动的纯共享网络研究

## 1. 结论与实际耗时

**纯共享神经补偿有明确继续研究价值。** 在固定Master、固定R3选择身份、相同K和12000 photo步下，SpatialFiLM相对等预算59D参照，fr1提升 **{f1['seven']-base1['seven']:.4f}dB**，fr2提升 **{f2['seven']-base2['seven']:.4f}dB**。所有候选补偿来自共享网络，没有Gaussian/Anchor私有可训练补偿参数。

最有价值的方向是**确定性物理输入编码＋连续神经场＋位移优化预条件**。输出节点插值是有用对照，但没有在两张地图、不同选择集合与长程训练下统一优于连续FiLM。没有找到稳定全面超过R3的神经架构，不能把一个平均值接近当成已经解决问题。

Anchor单位位移在fr2两seed的七档均值为 **{local2['seven']:.4f}dB**，同集合R3快照为 **{r32['seven']:.4f}dB**，差 **{local2['seven']-r32['seven']:+.4f}dB**；fr1仍有明显差距。连续FiLM在fr1总36000步为 **{longrows[1]['seven']:.4f}dB**，相对等36k的59D参照高 **{longrows[1]['seven']-longrows[0]['seven']:.4f}dB**，距12k R3快照 **{r31['seven']-longrows[1]['seven']:.4f}dB**。这是不同训练预算下的潜力观察，不是公平超越R3。

真实启动：2026-10-27 09:52:45香港时间；本报告生成时经过 **{elapsed:.1f}分钟**。预算180分钟；已有构建和较高实验吞吐使配对12k、36k续训、另一地图和机制检查提前完成。收尾依据是这些正式证据，而不是完成了多少短程作业。当前已完成{len(completed)}个训练运行、57个正式模型评价；其余为明确标记的短程或无效/未启动记录。本轮不做在线生产集成。

## 2. 实际边界、源码与参照

本地HEAD `427869a8421c5eb8d823827a19dfdf9c07f996b4`，分支anchor-si，启动tracked/staged干净。生产仍为Variable Anchor、正确scale/rotation split、S00R00E50D12；TF/SI/GI/ChangeDistill关闭，joint。服务器核心schema较本地旧，使用实际源码SHA与现有Renderer，不把本地Git当作已同步重建的服务器二进制。服务器172.19.163.58、conda zwj、Torch2.3.1/cu118、RTX3090；另有其他用户GPU任务，未干扰。

{markdown(topology,[('scene','冻结地图'),('N','Gaussian'),('M_stored','存储Anchor'),('occupied_anchors','有成员Anchor'),('empty_anchors','空Anchor'),('max_occupied','最多成员')])}

fr1沿用N6838地图、54个Bank训练/13个内部留出/45个历史non-Bank；fr2为N9275、47/12/64。训练320×240、评价640×480。两场景均重新训练各自网络，是另一地图的拟合确认，**不是同一权重零样本跨场景迁移**。历史视图已被多轮研究用过，不称全新盲测。

训练ratio=1/5/10/15%，未见=2.5/7.5/12.5%。fr1七档K=68/170/341/512/683/854/1025。same-K按`logit(p)-logit(cached_uniform)`排序，非纯概率Top-K。训练固定这些same-K集合；原Cached集合另评，实际K不同，不能把两策略混成同预算结果。

本轮主对照固定R3终点集合。Reference59从零补偿学习，保留历史Anchor/ratio latent表，仅作明确例外的参照，不作为候选。早期冻结其随机Encoder，随后增加可训练补偿Encoder控制，结论采用后者。实际Selector和全部身份始终冻结，不通过选择变化取巧。

R3参照来自之前完整辅助系统的终点：fr1 source3100、fr2 source4100。它们的训练过程中Selector曾变化；本轮没有重新优化个体补偿。故它是同地图/同最终集合的**快照质量参照**，不是相同固定集合训练协议的理论Oracle上界。用户提供的历史四seed/两seed系统均值16.9632/17.9932等不能直接与本轮固定集合均值相减。

## 3. 研究如何由发现推进

| 发现 | 实际后续决策 | 结果边界 |
|---|---|---|
| 物理编码、FiLM、空间编码短程优于RawPure | 提前做两seed完整12k，保留同预算59D | 正式收益存在，但seed分化明显 |
| 输出节点插值短程有竞争力 | 与连续FiLM配对，锁定fr2和原Baseline59集合 | 没有统一胜出，避免只选fr1最好结果 |
| 局部Attention与空间编码叠加不稳定 | 扩展一次全地图donor消息及moment/零moment控制 | 没有超过主线的明显信号，停止更大Attention |
| Teacher参数拟合改善未转成图像优势 | 4k监督＋8k photo对照4k photo＋8k photo重启 | 没有优势，不追加Teacher Loss |
| 12k两个seed分化 | 恢复完整Adam与帧序，匹配续训至36k；同seed复验 | 优化充分性影响很大，不能把短程差距当表达上限 |
| 节点共用Linear的Adam有非活跃更新 | 同参数/同函数拆分末层，真实photo验证clock | 拆分反而较弱，动量隔离不是自动有效的改进 |
| 位移单位影响弱seed | 检验Anchor单位的解析梯度，做fr2小规模确认 | 有优化预条件信号，尚不足以称普适提升 |
| SIREN在统一LR下不稳定 | 仅补一个有依据的1e-4控制 | 短程可恢复训练，但未超过主线，不扫激活/深度 |

另实现了对称仿射协方差输出`C'=LCLᵀ`，只用现有Renderer接口；零补偿图像最大误差约3.2e-6、真实photo梯度非零。4000步未超过主要控制，没有继续扩大。全部正负方向与执行状态在[NEURAL_V1_EXPERIMENTS.csv](NEURAL_V1_EXPERIMENTS.csv)，不把短程Bank最佳值当正式质量确认。

## 4. 纯网络数学形式与参数清单

只读输入19维：归一化世界XYZ3、asinh Offset3、log Scale3、SH DC3、Opacity logit1、trace归一化协方差对称6维。再加固定六频带世界XYZ正弦/余弦36维，共55维。没有训练空间网格，所有频率固定。输入人口统计冻结，不读取图像/相机或R3残差。

SpatialFiLM用共享55→64→64主干、共享log-ratio调制、XYZ/Scale/Rotation/Other四Head生成14D。SpatialLocalXYZ仅将位移输出乘原Anchor参考尺度；其余与FiLM相同。SpatialOutputKnots由共享网络预测四个ratio节点函数，再连续插值输出；节点值是本次Forward的激活，不是存储的个体可训练参数。

主候选参数分别为33806/33806/36536个，与N/M无关。完整列表见[NEURAL_V1_TRAINABLE_PARAMETERS.csv](NEURAL_V1_TRAINABLE_PARAMETERS.csv)及每模型`parameter_audit.json`。所有候选参数只属于共享Linear权重/偏置；没有Embedding、ID参数表、私有latent或可训练场景buffer。两地图参数shape一致，checkpoint只含这些共享参数及其Adam状态；Teacher输出和评价补偿保存在独立只读证据文件，不进入候选state_dict。

R3 Selector原本的Anchor特征属于冻结选择器，不进入本轮纯Compensation网络。Reference59表结构只作为参照，`pure_shared=False`明确标记，不能将它或R3/Factored包装成本轮候选。具体层次、公式、输出变换、冗余常数FiLM及部署边界见[NEURAL_V1_BEST_ARCHITECTURE.md](NEURAL_V1_BEST_ARCHITECTURE.md)。

## 5. 12000步正式质量与R3剩余差距

以下均为独立视图、same-K、两个优化顺序seed均值。它们不是两个独立SLAM地图seed；不选择最好重复。

{markdown(rows,[('scene','地图'),('name','补偿方式'),('train','训练ratio'),('unseen','未见ratio'),('seven','七档PSNR'),('p1','1%'),('p5','5%'),('ssim','SSIM'),('lpips','LPIPS'),('photo','photo')])}

低ratio依然有明显限制，不能只看七档均值。逐ratio指标与实际selected/visible/effective_proxy数量见`quality_by_ratio.csv`；逐视图mean/median/p10/p90和改善比例见`paired_results.csv`。visible是radii>0，effective_proxy再加opacity≥1/255，不是逐像素积分的真实贡献。

连续FiLM和输出插值在Bank训练及non-Bank均有收益，但没有证明所有ratio更好。输入物理编码、全局共享参数量、非线性和优化路径同时改变，不能把全部收益单独归于FiLM或世界坐标。相比59D，候选神经权重更多、整体可学习补偿参数更少，因为消除了私有表；这两点必须同时说明。

### 原Baseline59集合上的确认

{markdown(baserows,[('name','方式'),('train','训练ratio'),('unseen','未见ratio'),('seven','七档PSNR'),('ssim','SSIM'),('lpips','LPIPS')])}

同一神经架构在另一固定选择集合上重训，仍不改变K。连续FiLM的改善比输出插值更一致，说明R3集合上的结果不能直接推广到所有选择器；这也是下一轮保留连续条件化的重要依据。

该集合的2.5%是明确反例：两seed non-Bank均值，连续FiLM为 **15.3783dB**，输出插值为 **13.0593dB**，后者低 **2.3189dB**。这种局部退化会被七档平均掩盖，不能只报告总体均值。

## 6. 优化充分性与重复波动

以下总预算36000：从12k精确恢复Adam及Bank帧序，再24k真实photo步。不是丢弃动量重启，也没有伪造零梯度step。

{markdown(longrows,[('name','总36k方式'),('train','训练ratio'),('unseen','未见ratio'),('seven','七档PSNR'),('p1','1%'),('p5','5%'),('ssim','SSIM'),('lpips','LPIPS')])}

连续FiLM弱seed获得明显恢复，强seed的non-Bank则基本持平；不能解释成“更多训练总会提高泛化”。输出插值的训练节点改善与未见节点退化同时出现，故仅看Bank或节点PSNR会产生错误判断。原59D继续训练没有缩小全部差距，支持输入/共享组织确有价值，但不证明MLP的理论表达极限。

{markdown(repeats,[('name','seed5100方式'),('first','首次'),('repeat','同seed重跑'),('delta','重跑变化dB')])}

本轮同seed最大变化约 **{maximum_repeat:.4f}dB**。这不是通用噪声阈值；小于当前重复波动的机制差异不能认定稳定改进。正式主表保留原两seed，重复单列，不把5100第二次当独立seed，也不挑较好的一次。

## 7. 机制归因与失败启发

{markdown(controls,[('name','对照实验'),('base','参照'),('delta','non-Bank七档差'),('train_delta','训练ratio差'),('unseen_delta','未见ratio差')])}

输入屏蔽保持尺寸和网络容量不变。NoWorld同时屏蔽世界XYZ与固定空间频率，NoShape屏蔽协方差6维，NormalizedLocalOnly再去掉两者；这些控制支持物理/空间组织有作用，但幅度受seed与优化波动影响，不能把每个小差值写成稳定因果效应。

节点末层拆分在非零函数下逐值一致，真实photo两步验证：密集节点有非活跃Adam动量更新，拆分节点无此更新。质量没有相应提高，所以不能仅凭梯度隔离的理论动机加入新算法。常数FiLM折叠保持初始函数，却改变优化参数化，也没有改善。

Anchor单位满足`dL/dnetworkXYZ_i = anchor_scale_i × dL/dworldXYZ_i`，解析VJP相对误差小于3e-6。参考尺度按点有0.05/0.1等值，作用不是单一全局LR变化。两seed平均有正向信号，但尚缺独立新地图与更充分重复来确认优于所有世界单位设置。

Teacher的物理13D回归处理了Quaternion符号与SO(3)零点。早期一个零点梯度实现错误已在完成前中止，旧记录标为无效，修复后另目录重跑并通过真实梯度gate；无效作业不用于算法结论。修正后的4k监督＋8k photo仍未优于相同总更新次数的photo重启控制，不能据参数MSE下降声称图像收益。

SIREN统一1e-3 LR不稳定，1e-4能恢复短程训练；没有完成同预算长程确认，不能据此否定全部周期网络。局部Attention、全地图消息、moment和covariance输出的短程失败也只约束本轮具体实现，不能证明这些信息无研究价值。

### 事后Bank诊断：未训练端点的函数外推

正式图表暴露2.5%反例后，补做只读条件替换，不训练、不新增候选、不根据结果更改终点。固定网络、Master、K和Gaussian身份，对未见ratio的正常输出插值，分别仅将“未在低端点集合训练”的点改读高端点神经输出，以及对“已在低端点训练”的点做相同替换。以下为13个Bank留出视图、两seed均值，**不是non-Bank泛化结果**：

{markdown(endpoint_rows,[('source','固定选集'),('ratio','ratio'),('normal','正常插值'),('unsupported','仅替换低端点未训练点'),('supported','仅替换低端点已训练点'),('delta','未训练点替换差dB')])}

Baseline59集合2.5%的两个seed分别改善 **3.3696/2.2166dB**；R3集合分别改善 **0.5597/0.8932dB**。同一替换对已训练点没有相同收益，R3集合甚至下降。因此有实际条件干预证据支持：在本轮输出节点网络中，将未训练低端点的外推值混入未见ratio，会造成明显质量损失。不是仅凭输出平滑性或曝光相关性猜测。

但两个集合在2.5%都只有68/170点属于低端点，且高端点几乎完全覆盖；**总覆盖比例近乎相同，不能解释两个集合之间的退化幅度差异**。Gaussian具体空间/属性、端点函数外推和共同适配仍可能共同决定结果。替换点数不同，图像作用也非线性，不将两组差值相加或当作单位点贡献。此诊断没有独立新地图/新视图确认，也没有学习支持感知路由；不能把它包装成已验证的新算法。原始逐视图数据见`endpoint_support_probe.csv`，覆盖统计见`endpoint_selection_coverage.csv`。

## 8. 开销、显存和工程边界

{markdown(cost,[('name','原型'),('parameters','模块参数'),('params_mib','参数MiB'),('adam_mib','Adam MiB'),('forward_ms','Forward均值ms'),('backward_ms','Forward+Backward均值ms')])}

微基准在本轮正式作业结束后执行，仍有其他用户GPU任务。它包含当前Python输入构建、编码和补偿；Reference59 helper还计算N行Encoder，而神经候选为K行编码。因此这些是原型实际调用成本，**不是已优化生产Head的严格同范围速度排名**。峰值增量与各K见`mechanism/timing.csv`；不是整个Mapping显存或吞吐。

参数减少没有自动转成更快Forward：小网络的Python/Tensor创建、Fourier和调制会产生多个kernel。本轮暂不优化CUDA，也不把数MB辅助状态节省宣传成整张地图显存大幅下降。确定性输入与邻域缓存是只读数据，成本另列；主候选不依赖kNN消息缓存，在线还必须验证Master变化时的输入更新和归一化参考。

## 9. 证据强度与尚未解决问题

- **已验证：** 纯共享参数限制成立；固定集合下多个新神经场优于匹配59D；连续条件化与输出插值的差异依赖集合/地图/预算；只读Bank条件替换定位了未训练低端点外推的实际损害；长程优化与位移单位会改变结果；没有稳定全面超越R3。
- **合理假设：** 更可区分的物理输入与优化预条件共同改善学习效率；连续ratio共享可能减轻缺乏端点训练覆盖时的外推问题。没有从cosine或平滑性直接推断质量因果。
- **未确定：** 新网络的理论容量上限、各输入块收益的完全分离、最优训练机制，以及动态在线地图的泛化。没有测试15%以上ratio、真实SLAM、Full竞争或ATE。
- **版本记录局限：** 初批探索SHA在进程结束时读取文件，期间新增实验路径导致精确loaded版本记录不完整；后续改为import时SHA并保留快照。首批已配置路径的计算语义未改变，但不把该记录当精确loaded源码证据。训练/模型配置、原始日志和终点保留。

## 10. 交付与下一步

研究建议优先级：**用几何位移预条件缩短连续共享神经场达到高质量的训练路径 → 解释未见ratio的训练覆盖 → 再做有明确假设的局部交互**。不建议根据本轮结果直接加深/加宽网络，也不建议退回私有补偿表。

推荐下一轮检验：Anchor单位的连续网络能否在相同12k预算内，稳定达到本轮36k世界单位的质量；在新的地图锁定验证。需要同时改善1%/5%、超过重复波动、保留纯共享参数限制。具体可证伪方案见[NEURAL_V2_RESEARCH_PROPOSAL.md](NEURAL_V2_RESEARCH_PROPOSAL.md)。

状态：原型与固定地图实验 **IMPLEMENTED / TESTED**；微基准和局部机制 **PARTIALLY VERIFIED**；在线Joint、C++/CUDA、动态拓扑/闭环及生产替换 **NOT IMPLEMENTED**。生产代码与配置未改，不stage、commit或push。源码、配置、57个正式终点、逐视图证据、无效记录和SHA均保留，复现说明见[REPRODUCE.md](REPRODUCE.md)。
'''
    (ROOT/'NEURAL_V1_RESEARCH_REPORT.md').write_text(text,encoding='utf-8')
    state.update(phase='delivery',elapsed_minutes=elapsed,completed_runs=len(completed),formal_models=57,running=[],
        key_finding='pure shared physical neural fields improve matched59D; continuous and localunit optimization promising; no universalR3win',production_changed=False)
    (ROOT/'research_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
    print('REPORT_WRITTEN',len(text),elapsed,flush=True)

if __name__=='__main__':main()
