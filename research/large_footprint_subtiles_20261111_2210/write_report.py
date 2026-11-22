"""从冻结实测数据生成中文报告，保留全部反例与精度边界。"""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'reports/AGGREGATE'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(x) for x in row)+'|' for row in rows])


def main():
    metrics=pd.read_csv(DATA/'CONDITIONS.csv')
    attr=pd.read_csv(DATA/'ATTRIBUTION_SUMMARY.csv')
    summary=json.loads((DATA/'SUMMARY.json').read_text(encoding='utf-8'))
    full=pd.read_csv(DATA/'PER_SCENE_RATIO_ROUND.csv')
    failures=full[(full.variant=='common')&(full.condition=='full')&(full.speedup<1)]
    scaling=pd.read_csv(DATA/'COUNT_SCALING_SUMMARY.csv')
    profiles=pd.read_csv(HERE/'reports/PROFILE/SUMMARY.csv')
    modules=pd.read_csv(HERE/'reports/PROFILE/MODULES.csv')
    freeze=json.loads((HERE/'reports/FROZEN.json').read_text(encoding='utf-8'))
    integrity=json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text(encoding='utf-8'))
    cleanup=json.loads((HERE/'reports/CLEANUP.json').read_text(encoding='utf-8'))
    assert summary['bitwise_pass'] and integrity['passed']
    lines=[]
    def add(text):lines.append(text.strip()+'\n')
    add('''# 低 Ratio、大 footprint CUDA 渲染优化报告

本轮结论：**没有找到同时具备实质低 Ratio 提速、逐组 Full 不退化、RGB/radii 逐位一致的新增通用方案。最终推荐继续使用原 F2，生产默认不变。**

精确公共乘积候选在32组正式双轮测量中相对原F2约快0.4%–0.9%，但追加同二进制原逻辑对照后，1%下略慢，5%/10%的增量仅约0.07%/0.11%；Full还有4组未通过两轮均不退化门限。不能把这组微小差异解释为解决了大footprint问题。

直接Tile成员遍历在超大footprint、强遮挡的合成压力输入上最高8.31×，真实场景却明显退化，因此不作为通用方案。所有失败候选与反例均保留。

## 1. 范围、对照和冻结条件

- 仅优化训练结束后的Rasterizer，准备、Selector、Compensator、Gaussian身份和补偿后属性不变。没有训练、重新选点或修改checkpoint。
- 对照F2是本轮开始时已验证的优化渲染入口，位于`render_count_scaling_20261111_2022/background_stage`。本报告所有增量均相对该入口，**不重复计算前轮收益**，也不将其称为原始CaRtGS速度。
- 51个模式ID包含控制模式0；46个候选进入8轮有效独占开发筛选。主候选49 `exact_common32`与比较候选18 `warp_exit8`在正式验证前冻结，不使用保留集调参。
- 开发集4组：Replica office2 RGB-D、room0 RGB-D，TUM fr1_xyz RGB-D，VECtor corner-slow stereo。其余28组为保留验证。
- 不改变Alpha阈值、透射率早停阈值、深度排序/同深度顺序、SH、分辨率、精度、合成递推顺序和背景语义。不按Ratio/K阈值让Full绕开新路径。

服务器研究目录：`/home/prmi/zwj/experiments/large_footprint_subtiles_20261111_2210`。
本地对应目录：本报告所在目录。生产路径、共享conda和模型未写入；未commit、push或部署。

## 2. 正式测量协议与覆盖

RTX 3090，CUDA 11.8，现有zwj环境。Replica全部16个场景/模式、TUM10组、VECtor6组，共32组；地图N范围4,644–79,351。输入checkpoint、原图相机和前轮共享模块来自冻结目录，明细见`reports/INPUT_CATALOG.json`与`INPUT_SHA256.json`。

每组12条件：局部Ratio `.01,.03,.05,.1,.2,.3,.4,.5,.7`，固定预算1000/5000，以及全图Full。局部预算以Viewer名义Anchor候选池的当前N计算严格K；Full输入全图。因此局部Ratio与全图K/N不是同一个量，不能把局部70%直接连成全图Full的同域曲线。

固定预算也沿用原入口的候选不足处理，实际K可能小于请求预算；每帧K均保留，不将请求1000/5000误记为所有视角实际值。

每条件6个连续视角、4个跳转视角，每视角10次重复，2轮独立进程；第二轮反转组序/视角/条件，版本次序均衡轮换。输入在计时外按当前视角重新生成，计时读取相同Gaussian与补偿后属性；没有缓存上一帧预测或RGB。预编译CUDA Graph只是形状计划复用，不是同帧结果复用。

主口径是**纯Rasterizer的同步墙钟**；CUDA Event另测；Nsight Kernel累计时间另表，不混用。准备、网络与完整SLAM端到端时间本轮未测，因此不宣称端到端加速。

正式采集230,400条墙钟、46,080条CUDA Event、23,040条逐位比较、3,840条计时外全像素负载诊断、192条Workspace记录。32个有效测试段均无监测到的外部GPU/CPU活跃污染；7段被污染而整段重跑，原记录保留。追加归因采集345,600条墙钟和11,520条逐位比较，5段污染重跑。

监控采样中GPU温度52–74℃，SM时钟345–1980MHz，显存总使用565–2718MiB；这些范围包含加载/空闲/预热，不能视为计时区间锁频。微秒级小差异仍可能受DVFS和代码布局影响，不能仅以均值正收益认定成功。具体`*_MONITOR.jsonl`随证据保留。

## 3. 正式结果：原F2与精确公共乘积

下表p50/p95为各场景各轮分位数的中位数；加速比为逐场景逐轮配对加速比的几何平均，两个聚合口径不能相除互推。单位μs。''')
    rows=[]
    order=['local0.01','local0.03','local0.05','local0.1','local0.2','local0.3','local0.4','local0.5','local0.7','fixed1000','fixed5000','full']
    for condition in order:
        base=metrics[(metrics.condition==condition)&(metrics.variant=='F2')].iloc[0]
        c=metrics[(metrics.condition==condition)&(metrics.variant=='common')].iloc[0]
        rows.append([condition,f'{base.p50_ms*1000:.3f}',f'{c.p50_ms*1000:.3f}',f'{base.p95_ms*1000:.3f}',
                     f'{c.p95_ms*1000:.3f}',f'{c.speedup:.6f}×',f'{int(c.faster_both_rounds)}/32'])
    add(table(['条件','F2 p50','候选p50','F2 p95','候选p95','配对加速','两轮均不慢组数'],rows))
    add('''Full平均略快，但门限要求每一组两轮均不慢，实际仅28/32组满足。以下全部反例不作剔除，不修改门限。''')
    add(table(['场景/模式','轮次','F2 p50 μs','候选p50 μs','延迟变化','绝对增加 μs'],[
        [r.scene,r['round'],f'{r.F2_p50_ms*1000:.4f}',f'{r.p50_ms*1000:.4f}',f'{(1/r.speedup-1)*100:+.4f}%',f'{-r.saved_us:.4f}']
        for _,r in failures.iterrows()]))
    add('''两轮变化方向不一致的组也视为未通过；TUM fr1_room mono两轮都慢。逐场景逐Ratio所有p50/p95、K、重复波动见`PER_SCENE_RATIO_ROUND.csv`；所有慢例见`ALL_SLOWER_SCENE_ROUNDS.csv`。28组保留集同样只有小幅平均差异，未提供实质加速证据。

## 4. 追加归因：同二进制原逻辑对照

正式结果小于1%，而开发筛选中重新编译的F2控制逻辑也出现相近变化，因此增加原F2、同二进制`Renderer(0)`、同二进制候选49三方配对。候选不再改变；增加30次重复、双轮、全部32组和相同10个移动/跳转视角。该试验用于归因，**不覆盖或挽救第3节失败的门限**。''')
    add(table(['条件','重新编译原逻辑/F2','公共乘积/F2','公共乘积/同二进制原逻辑','实际新增延迟降低'],[
        [r.condition,f'{r.rebuilt_vs_F2:.6f}×',f'{r.common_vs_F2:.6f}×',f'{r.common_vs_rebuilt:.6f}×',
         f'{100*(1-1/r.common_vs_rebuilt):+.4f}%'] for _,r in attr.iterrows()]))
    add('''结论：不能把相对旧F2的全部微小差异归于FMA复用；同二进制控制消除了大部分优势。Nsight中像素Kernel寄存器由48降至40，shared仍为9216B、block仍16×8，但寄存器减少没有转化为有实用价值的完整Rasterizer收益。

## 5. 大footprint为什么仍然贵

计时外诊断保持原Alpha阈值和T早停，只统计真正写入颜色的Gaussian/像素交互；诊断RGB与原输出逐位核对。以下每一列均为逐组比值的中位数，不是各列总量比。''')
    rows=[]
    for condition in ['local0.01','local0.05','local0.1','local0.3','local0.5']:
        r=scaling[scaling.condition==condition].iloc[0]
        rows.append([condition,f'{r.K_vs_full*100:.2f}%',f'{r.interaction_vs_full*100:.2f}%',
            f'{r.instances_vs_full*100:.2f}%',f'{r.latency_vs_full*100:.2f}%',f'{r.active_tile_fraction*100:.1f}%'])
    add(table(['局部Ratio','输入K/Full','Alpha交互/Full','Tile实例/Full','F2延迟/Full','活跃Tile占比'],rows))
    add('''例如局部1%时，输入K仅约全图的0.33%，真正Alpha交互仍约Full的33.49%，延迟仍约35.04%；局部10%时，K约3.34%，Alpha交互仍69.84%。这不是额外枚举空区域就能消除的负载。

子块筛除在4个开发场景、连续/跳转各一视角、5种条件、4/8像素两种子块共80条诊断上，检查87,738,680个区域，逐像素误拒绝为0。**1%时4×4仅筛除2.51%，8×8仅1.21%。** 大footprint内部多数子块真的存在贡献，额外构建掩码、压缩和读取成本超过省下的工作。

在固定图像尺寸和逐位语义下，至少需要输出HW像素；FP32的T和颜色递推也不能任意改为并行树归约。当前计时更接近固定成本＋实际遍历/Alpha交互成本，而非只由输入K决定。数据支持“减少点数通常能降低延迟”，**不支持K与延迟斜率为1或严格比例关系**，也不证明每个视角均单调。

## 6. 实现、失败候选及回退

所有实现均在独立研究目录，模式可单独选择；未修改生产入口。原F2保留在前轮目录，最终推荐记录在`reports/FINAL_SELECTION.json`。''')
    add(table(['候选族/模式','实现与动机','实测结论/处理'],[
        ['4×4/8×8子块，1–4','保守区间、舍入外扩；不确定则保留。掩码与压缩两种路径','零误筛，但1%开发加速比约0.806×/0.929×；放弃'],
        ['批量Alpha，5–10','批量预计算exp/alpha，尝试用寄存器摊薄分支和重复读取','寄存器/额外工作抵消收益；5–7在1%约0.897–0.942×；放弃'],
        ['跨Tile共享，11–14','2×1/2×2宏Tile共享成员，减少重复枚举','并集成员和分支扩大；1%约0.912–0.955×；放弃'],
        ['早停，15–19','逐线程退出或每2/4/8/16成员warp检查T完成','18开发看似约1%收益，正式1%约1.00324×，其余多数近持平或慢；放弃'],
        ['直接成员遍历，20–21','按深度批次过滤当前Tile，用CUB BlockScan稳定压缩后合成，绕过位表/compact','真实5%约0.883×，Full约0.344×；仅合成强遮挡有大收益，未采用'],
        ['short/tile cache，22–33','32/64/128条shared缓存、像素布局、unroll等','多数近持平或退化，无稳定普遍收益；放弃'],
        ['shared Alpha，34–39','共享批量alpha与转置布局','额外shared与同步开销；1%约0.668–0.919×；放弃'],
        ['occupancy，40–43','launch bounds、寄存器上限、组合早停','1%约0.984–1.003×，无实质收益；放弃'],
        ['compact+pixel，44–46','把压缩与合成放在同一CTA，减少一次Kernel','同步与CTA成本抵消launch节省；1%约0.985–0.995×；放弃'],
        ['精确公共乘积，47–50','共享两像素的dx及相关乘积，显式保持实际SASS的rn/FMA方向','49逐位通过，同二进制新增收益极小，Full门限失败；仅研究保留']]))
    add('''直接遍历初版`validation06`发生非法访问，原因是二维block与`cub::BlockScan<int,128>`的一维线程编号不匹配。修复为一维128线程，使用threadIdx.x映射像素；`validation07`752项通过。失败源码/日志保留，旧等待器的虚假waiting状态已核对进程退出后更正为failed，未重复启动。

普通代数公共项提取可能改变FMA融合和舍入，不能直接使用。本轮精确候选根据F2实际SASS保留计算：

```text
dx      = rn(mean.x - pixel.x)
ax      = rn(co.x * dx)
cross_x = rn(co.y * dx)
yy      = rn(dy * rn(co.z * dy))
cross   = rn(dy * cross_x)
sum     = fma_rn(dx, ax, yy)
power   = fma_rn(sum, -0.5, -cross)
```

只共享两像素的dx/ax/cross_x，不改变exp、Alpha/T和颜色累加。依据是当前RTX3090编译产物，不声称其他编译器/架构无须复验。

## 7. 合成压力的适用边界

48种人工构造输入：641×479或1280×720，K=1/32/128/512，Scale=.03/.3/3，Opacity=.02/.5；每条件20次轮换重复。五个版本共240项图像/radii逐位比较，4800条计时。

1280×720、K=512、Scale=3、Opacity=.5时，直接成员遍历由846.492μs降至101.872μs，8.31×；同条件641×479为5.17×。强遮挡使得合成早期结束，直接遍历可以不再完整建立后方成员结构。

但这不是当前真实LoD的普遍负载。对所有Tile重新扫描全局深度列表在稀疏/较大K时浪费大量工作；真实5%/10%/Full明显更慢。即使用户允许Full慢，也没有证据支持它在真实低Ratio上取得显著通用收益，不能据此部署。

精确公共乘积在合成48条件上加速比范围0.947–1.055×；存在约5.6%延迟退化的压力反例。压力实验与真实模型性能表始终分开。

## 8. Nsight、CUDA Event与Host等待

Nsight只采office2 RGB-D和corner-slow stereo，两移动/跳转视角、5条件、3版本、6次重复，360次调用。各版本使用相同Graph节点组织，**没有减少Kernel Launch或同步次数**。低Ratio通常7–8个Kernel；Full为office2的14个、corner-slow的11个。每次调用仍有一个输入更新Kernel、一个Graph Launch；测量显式同步一次。

Profile监测无外部活跃GPU，但6个采样中有2个外部CPU活跃样本，故仅用于GPU热点归因，不代替有效独占墙钟。原始`.nsys-rep`保留，SQLite可由其重建；原API与Kernel表均保留。''')
    rows=[]
    for scene in ['Replica__office2__rgbd','VECtor__corner-slow__stereo']:
        for condition in ['local0.01','local0.1','full']:
            r=profiles[(profiles.scene==scene)&(profiles.condition==condition)&(profiles.variant=='F2')].iloc[0]
            pix=modules[(modules.scene==scene)&(modules.condition==condition)&(modules.variant=='F2')&(modules.module=='pixel')].iloc[0]
            rows.append([scene.replace('__','/'),condition,int(r.kernels),f'{r.gpu_us:.2f}',f'{pix.us_per_render:.2f}',f'{100*pix.us_per_render/r.gpu_us:.1f}%'])
    add(table(['场景','条件','Kernel数','Kernel累计μs','像素合成μs','合成占比'],rows))
    add('''Kernel累计、CUDA Event和同步墙钟都是不同口径。Event配对结果见`CUDA_EVENTS.csv`：候选1%约1.00197×、5%1.00890×、10%1.00507×，20%/40%略慢，亦不支持全负载稳定提速。不可从Profile的像素阶段下降直接推导完整SLAM加速。

## 9. 正确性、内存与生命周期

- 初筛`validation13`1504项逐位通过；正式23,040项与追加11,520项比较均RGB/radii bitwise一致，最大RGB绝对差0。
- 格式：SH/预计算颜色、scale/quaternion/预计算covariance，非整Tile尺寸、极扁/极大尺度、同深度、空集合、非有限输入和Alpha边界。
- 最终候选额外172项生命周期检查：K在31/32/33、255/256/257、511/512/513、1023/1024/1025等边界；短列表溢出回退，Tile数量255/256/257；多Stream显式事件依赖及错误实例跨Stream拒绝；连续变化形状不插同步，保留所有输出核对独立性。
- Compute Sanitizer `memcheck`、`initcheck`、`synccheck`各自完整172项，均`ERROR SUMMARY: 0 errors`。未运行racecheck，不声称覆盖所有并发错误类型。
- 原有最多512个形状Graph计划、Workspace容量限制和LRU保留，实例绑定device/stream。参数/输入指针每帧更新，形状改变按原机制预分配或重建；没有跨视角预测缓存。
- 精确公共乘积和早停候选不增加Workspace：与F2各组逐项相同，为5.535–53.712MiB。该数是预分配Workspace，**不是全进程显存峰值**。监控的全GPU显存范围包含驱动/其他常驻context，不作为候选独占峰值。
- 仅推理路径研究；没有训练或Backward变动，不对新训练行为作未测试声明。

## 10. 完整性与文件管理
''')
    add(f'''最终二进制：`{freeze['binary']}`。
SHA256：`{freeze['binary_sha256']}`。

完整性终检通过：72项前轮基线源码/二进制、179项冻结checkpoint/输入及46项本轮源码/入口文件。三个控制器全部complete，PID/start_ticks核对均已退出；无本轮GPU任务遗留。验证范围是清单中的文件，不是对服务器所有生产文件的整体哈希审计。

仅清理本轮{len(cleanup['files'])}个可重建产物，共{cleanup['total_bytes']/(1<<20):.2f}MiB：废弃候选so、两个编译对象、可重建SQLite、已提取的中间传输包和被最终归档替代的草稿包。删除前核对真实路径在本轮目录、非软链接、无存活进程映射；原F2与最终完整so、失败源码快照、Profile、日志和全部数值保留。逐文件SHA与清理原因见`reports/CLEANUP.json`。未触碰模型、数据集、共享缓存或其他任务目录。

没有保存渲染图片。`figures/`只有性能/负载曲线PNG与SVG，非场景RGB渲染。
''')
    add('''## 11. 复现入口与交付

沿用现有服务器环境；不要覆盖已有结果标签。示例在空闲GPU上以新的输出目录运行：

```bash
cd /home/prmi/zwj/experiments/large_footprint_subtiles_20261111_2210
PY=/home/prmi/anaconda3/envs/zwj/bin/python
$PY -B launch.py review_bitwise $PY -u -B validate.py --output review_bitwise --modes 49,18
$PY -B launch.py review_lifecycle $PY -u -B lifecycle.py --output review_lifecycle --modes 49,18
$PY -B launch.py --measure review_pair $PY -u -B attribution.py --output review_pair --round 1 --group 0
```

`launch.py`拒绝重复存活任务和覆盖结果；返回只表示已启动，须核对`reports/jobs`的退出码及监控。测量必须空闲，不能直接重跑已有controller或覆盖accepted列表。

证据依赖当前服务器保留的只读前轮研究模块，包内的输入清单给出绝对路径和哈希；不是包含数据集/模型的独立可迁移镜像。

主要文件：

|文件|内容|
|---|---|
|`reports/AGGREGATE/PER_SCENE_RATIO_ROUND.csv`|全部场景、Ratio、两轮p50/p95/K/波动|
|`reports/AGGREGATE/PAIRED_SPEEDUP.csv`|逐组双轮配对与退化门限|
|`reports/AGGREGATE/ALL_SLOWER_SCENE_ROUNDS.csv`|所有退化项|
|`reports/AGGREGATE/ATTRIBUTION_*.csv`|同二进制控制与归因|
|`reports/AGGREGATE/*SAMPLES.csv.gz`及`samples.csv.gz`|逐次计时，不仅均值|
|`reports/AGGREGATE/COUNT_SCALING.csv`|实际K、Alpha交互、Tile工作量与Full比|
|`reports/AGGREGATE/SYNTHETIC_STRESS.csv`|合成压力性能与全部反例|
|`reports/PROFILE/`与`profile01.nsys-rep`|Kernel/Host API分解与原始Profile|
|`source/`、`source_*/`、`build/*.so`|最终实现与失败候选源码快照|
|`logs/`|编译、测试、Sanitizer、污染监控与退出证据|
|`figures/RESULT_OVERVIEW.*`|收益、工作量、子块有效性、控制归因|
|`figures/SCENE_LATENCY_01.*`、`02.*`|全部32组延迟曲线与N|
|`figures/SCENE_COUNT_LATENCY_01.*`、`02.*`|线性实际K与延迟曲线，不预设点数正比|
|`figures/KERNEL_BREAKDOWN.*`|实际Kernel耗时组成|
|`figures/SYNTHETIC_FOOTPRINT_STRESS.*`|合成强遮挡的适用边界|

## 12. 剩余优化空间

1. 更有价值的方向是**按实际遮挡提前截断后方成员构建**。直接遍历压力试验提供机制证据，但当前全局扫描成本不可接受。下一候选应使用Tile级有界分块，只在确定所有像素已满足原T阈值时停止，保持深度顺序及FP32递推，不用K/Ratio硬切分规避反例。
2. 另一个方向是减少真实Alpha合成所需的每成员指令，而不是继续更细地筛空区域。本轮SASS公共项已经验证可保持逐位，但收益不足；更激进的exp/递推变换会涉及数值顺序，应单独验收，不能默认纳入。
3. 输出HW像素和大面积真实贡献构成固定/面积成本。要使“点数减少10倍→延迟也减少10倍”，仅改变成员枚举通常不够；在不改变分辨率、Alpha语义和精度的约束下，本轮没有证据支持该承诺。

最终保留原F2为推荐版本，独立研究目录保留已编译候选和完整证据；不自动修改生产配置。
''')
    (HERE/'LOW_RATIO_LARGE_FOOTPRINT_OPTIMIZATION_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('CHINESE_REPORT_WRITTEN')


if __name__=='__main__':main()
