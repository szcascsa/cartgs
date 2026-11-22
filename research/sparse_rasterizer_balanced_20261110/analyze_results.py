"""从原始配对样本生成中文报告、逐场景CSV与静态性能图。"""
from pathlib import Path
import csv,json,statistics as st

HERE=Path(__file__).resolve().parent
BASE=HERE/'results/reports'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def csv_read(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile(values,q):
    values=sorted(values);i=(len(values)-1)*q;lo=int(i)
    return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(i-lo)


def aggregate(family,script,variants,field):
    rows=[]
    for j in [1,2]:
        path=BASE/family/f'{script}{j}'
        assert read(BASE/family/f'{script}{j}_RESULT.json')['foreign_active_samples']==0
        for r in csv_read(path/'samples.csv'):r['round']=j;rows.append(r)
    results=[]
    for scene,condition in sorted({(r['scene'],r['condition']) for r in rows}):
        group=[r for r in rows if (r['scene'],r['condition'])==(scene,condition)]
        result=dict(scene=scene,condition=condition,K_median=st.median(float(r['K']) for r in group),
            K_min=min(int(r['K']) for r in group),K_max=max(int(r['K']) for r in group))
        for variant in variants:
            g=[r for r in group if r['variant']==variant];v=[float(r[field]) for r in g]
            result[variant+'_ms']=st.median(v);result[variant+'_p90_ms']=quantile(v,.9)
            if field=='total_ms':
                result[variant+'_prepare_ms']=st.median(float(r['prepare_ms']) for r in g)
                result[variant+'_render_ms']=st.median(float(r['render_ms']) for r in g)
        for variant in variants:
            if variant!='balanced':result['speedup_vs_'+variant]=result[variant+'_ms']/result['balanced_ms']
        for j in [1,2]:
            by_round=[r for r in group if r['round']==j]
            for variant in ['original','bitset_v1']:
                a=st.median(float(r[field]) for r in by_round if r['variant']==variant)
                b=st.median(float(r[field]) for r in by_round if r['variant']=='balanced')
                result[f'round{j}_vs_{variant}']=a/b
        results.append(result)
    return results,len(rows)


def table(rows,fields):
    header='| '+' | '.join(label for key,label in fields)+' |\n'
    header+='| '+' | '.join('---' for _ in fields)+' |\n'
    for row in rows:
        header+='| '+' | '.join(f'{row[key]:.4f}' if isinstance(row[key],float) else str(row[key]) for key,label in fields)+' |\n'
    return header+'\n'


def figures(raster,pipeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    full=[r for r in raster if r['condition']=='full']
    fig,ax=plt.subplots(figsize=(12,9),layout='constrained')
    ys=list(range(len(full)))
    for dy,key,label,color in [(-.24,'original','原渲染器','#555555'),(0,'warp_v1','上一版 Warp','#4E79A7'),(.24,'balanced','新版有序列表＋目录','#E15759')]:
        ax.barh([y+dy for y in ys],[r[key+'_ms'] for r in full],height=.22,label=label,color=color)
    ax.set_yticks(ys,[r['scene'].replace('Replica/','')+f'  N={r["K_max"]:,}' for r in full]);ax.invert_yaxis()
    ax.set_xlabel('Full 纯渲染延迟 / ms（越低越好）');ax.legend();ax.grid(axis='x',alpha=.2)
    ax.set_title('全部 16 场景 Full 复验 · 同输入，FP32 RGB/radii 逐位一致\n两轮空闲 GPU 配对中位数；原分辨率，未改网络、点集或属性')
    fig.savefig(HERE/'FULL_RENDER_SPEED.png',dpi=160);plt.close(fig)
    scenes=['Replica/office1/mono','Replica/office2/rgbd','Replica/room0/rgbd','Replica/room2/mono']
    fig,axes=plt.subplots(4,2,figsize=(13,13),layout='constrained')
    ratios=[.01,.05,.1]
    for i,scene in enumerate(scenes):
        for j,data in enumerate([raster,pipeline]):
            group=[next(r for r in data if r['scene']==scene and r['condition']==c) for c in ['local1pct','local5pct','local10pct']]
            ax=axes[i,j]
            for key,label,color in [('original','原渲染器','#555555'),('bitset_v1','上一版 Bitset','#4E79A7'),('balanced','新版有序列表＋目录','#E15759')]:
                ax.plot(ratios,[r[key+'_ms'] for r in group],'o-',label=label,color=color)
            ax.set_xticks(ratios,[f'{v:.0%}\nK≈{r["K_median"]:,.0f}' for v,r in zip(ratios,group)])
            ax.set_title(scene+(' · 纯渲染' if j==0 else ' · 每帧重新准备＋渲染'))
            ax.set_xlabel('当前可见 Anchor 候选池预算 Ratio（线性轴）')
            ax.set_ylabel('延迟 / ms');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend();axes[0,1].legend()
    fig.suptitle('低 Ratio 真实推理 · 每次重新计算视角候选和网络输出\n同次序选点、同补偿结果；两轮配对中位数')
    fig.savefig(HERE/'LOW_RATIO_RENDER_SPEED.png',dpi=160);plt.close(fig)


def view_check():
    full=[];small=[]
    for family in ['formal','quiet']:
        for j in [1,2]:
            groups={}
            for row in csv_read(BASE/family/f'round{j}/samples.csv'):
                key=(row['scene'],row['condition'],row['frame'],row['variant'])
                groups.setdefault(key,[]).append(float(row['wall_ms']))
            for (scene,condition,frame,variant),values in list(groups.items()):
                if variant!='balanced':continue
                ms=st.median(values)
                if condition=='full':full.append(st.median(groups[(scene,condition,frame,'original')])/ms)
                if condition in ['local1pct','local5pct','local10pct']:
                    small.append(dict(family=family,speedup=st.median(groups[(scene,condition,frame,'bitset_v1')])/ms))
    assert min(full)>=1
    return dict(full_view_round_cells=len(full),full_min_speedup=min(full),
        quiet_low_ratio_view_regressions=sum(r['family']=='quiet' and r['speedup']<1 for r in small),
        formal_low_ratio_view_regressions=sum(r['family']=='formal' and r['speedup']<1 for r in small),
        formal_low_ratio_max_latency_increase_pct=max(0,100*(1/min(r['speedup'] for r in small if r['family']=='formal')-1)))


def main():
    assert read(BASE/'quiet/COMPLETE.json')['passed'] and read(BASE/'formal/COMPLETE.json')['passed']
    all_inputs=read(BASE/'ALL_RENDER_INPUTS_VALIDATION.json')
    assert all_inputs['passed'] and all_inputs['cases']==896
    workloads=[r.copy() for r in all_inputs['rows'] if 'tiles' in r]
    for row in workloads:row['compact_active_pct']=100*row['compact_nonempty_tiles']/max(1,row['active_tiles'])
    csv_write(HERE/'TILE_WORKLOADS.csv',workloads)
    raster,n=aggregate('quiet','round',['original','bitset_v1','warp_v1','balanced'],'wall_ms')
    pipeline,np=aggregate('quiet','pipeline',['original','bitset_v1','balanced'],'total_ms')
    auxiliary,na=aggregate('formal','round',['original','bitset_v1','warp_v1','balanced'],'wall_ms')
    views=view_check()
    full=[r for r in raster if r['condition']=='full']
    gate=all(r[f'round{j}_vs_original']>=1 for r in full for j in [1,2])
    assert gate,'预先声明的Full门限失败'
    csv_write(HERE/'RASTERIZER_RESULTS.csv',raster);csv_write(HERE/'PIPELINE_RESULTS.csv',pipeline)
    csv_write(HERE/'MONITOR_SENSITIVITY_RESULTS.csv',auxiliary)
    profiles=read(BASE/'formal/profile/SUMMARY.json')['results']
    csv_write(HERE/'GPU_PROFILE.csv',[{k:r[k] for k in ['condition','variant','K','cuda_event_ms','gpu_kernel_us','kernels_per_frame',
        'memcpy_per_frame','memset_per_frame','sync_calls_per_frame','sync_cpu_us_per_frame','workspace_bytes']} for r in profiles])
    small=[r for r in raster if r['condition'] in ['local1pct','local5pct','local10pct']]
    low_range={c:dict(min=min(r['speedup_vs_bitset_v1'] for r in small if r['condition']==c),
        max=max(r['speedup_vs_bitset_v1'] for r in small if r['condition']==c)) for c in ['local1pct','local5pct','local10pct']}
    psmall=[r for r in pipeline if r['condition'] in ['local1pct','local5pct','local10pct']]
    memory=read(BASE/'quiet/round2/SUMMARY.json')['workspace_bytes']
    integrity=read(BASE/'quiet/COMPLETE.json')
    facts=dict(full_gate_passed=gate,scenes=len(full),full_min_speedup=min(r['speedup_vs_original'] for r in full),
        full_max_speedup=max(r['speedup_vs_original'] for r in full),
        full_vs_warp_min=min(r['speedup_vs_warp_v1'] for r in full),
        full_vs_warp_max=max(r['speedup_vs_warp_v1'] for r in full),
        low_ratio_incremental_speedups=low_range,
        low_pipeline_min=min(r['speedup_vs_original'] for r in psmall),low_pipeline_max=max(r['speedup_vs_original'] for r in psmall),
        low_pipeline_increment_min=min(r['speedup_vs_bitset_v1'] for r in psmall),
        low_pipeline_increment_max=max(r['speedup_vs_bitset_v1'] for r in psmall),
        local1pct_saved_us_min=min(1000*(r['bitset_v1_ms']-r['balanced_ms']) for r in small if r['condition']=='local1pct'),
        local1pct_saved_us_max=max(1000*(r['bitset_v1_ms']-r['balanced_ms']) for r in small if r['condition']=='local1pct'),
        main_raster_samples=n,main_pipeline_samples=np,supplementary_raster_samples=na,
        workspace_bytes=memory,additional_workspace_bytes=memory['balanced']-memory['bitset_v1'],integrity=integrity,
        view_level_checks=views,all_render_input_cases=all_inputs['cases'],
        real_input_resolutions=sorted({(r['width'],r['height']) for r in all_inputs['rows']}))
    (HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    selected=[r for r in raster if r['scene'] in ['Replica/office1/mono','Replica/office2/rgbd','Replica/room0/rgbd','Replica/room2/mono'] and r['condition'] in ['local1pct','local5pct','local10pct','full']]
    doc=f'''# Full不退化的低Ratio CUDA渲染优化报告

## 结论与验收

新版通过预先声明的门限：全部16个Replica场景/相机模式、两轮独立测量中，Full中位延迟均不慢于原入口。Full加速为 **{facts['full_min_speedup']:.2f}–{facts['full_max_speedup']:.2f}×**，相对上一版Warp路径仍为 **{facts['full_vs_warp_min']:.2f}–{facts['full_vs_warp_max']:.2f}×**。

低Ratio相对上一版已验收Bitset的新增加速：1%为{low_range['local1pct']['min']:.3f}–{low_range['local1pct']['max']:.3f}×，5%为{low_range['local5pct']['min']:.3f}–{low_range['local5pct']['max']:.3f}×，10%为{low_range['local10pct']['min']:.3f}–{low_range['local10pct']['max']:.3f}×。这些是本轮新增收益；相对最原始渲染器的历史收益单列在CSV中。

4个代表场景每次重新切换视角、重新准备候选及Selector/Compensator输出后，1%/5%/10%的准备＋渲染相对原入口加速 **{facts['low_pipeline_min']:.2f}–{facts['low_pipeline_max']:.2f}×**。这不是完整SLAM/Mapping或端侧设备测速，也不含传输、Viewer显示和图像存盘。

上述准备＋渲染相对上一版Bitset的**本轮新增**为{facts['low_pipeline_increment_min']:.3f}–{facts['low_pipeline_increment_max']:.3f}×。1%纯渲染本轮每帧实际节省{facts['local1pct_saved_us_min']:.1f}–{facts['local1pct_saved_us_max']:.1f}μs，不能将相对最原始入口的数倍收益全部归给本轮。

## 实际实现

保留共享深度稳定排序和Gaussian→Tile位表，新增两项互补机制。第一项每32个位表字构建非空目录，像素只枚举非空字，不再逐字扫描整张地图的深度位表。第二项每个Tile用一个Warp按原深度顺序把成员压成最多128项的有序列表，列表内成员供全部像素复用；真实成员超过128时，按目录完整枚举，**不截断、不丢点、不改变Alpha顺序**。分支依据当前Tile实际成员数，未按Ratio或地图K让Full绕开新算法。

原像素FP32幂值、exp、Alpha阈值、透射率更新、早停及颜色累加表达式保持不变。非空输入的输出图像及radii改用empty，因为预处理及像素核会写全，减少无消费者的清零。所有Scratch逐帧重新构建，不缓存上一个视角的几何或网络输出；扩容不复制旧Scratch。返回图像独立持有存储，不被下一帧覆盖。

新增目录、成员列表和真实计数使用持久Workspace。实测跨16场景保留的最大工作区：新版{memory['balanced']/2**20:.2f}MiB，上一版Bitset {memory['bitset_v1']/2**20:.2f}MiB，增加{facts['additional_workspace_bytes']/2**20:.2f}MiB；上一版Warp {memory['warp_v1']/2**20:.2f}MiB。该数值是组件持久Workspace，不是整个应用峰值显存。位表单独上限256MiB；超限明确报错，不静默回退或无界缓存。每个实例绑定一个device/Stream。

## 为什么Full改善更明显

以真实1200×680输入的第19帧为例，office2 RGB-D Full位表每Tile长1350个32位字，实际平均仅39.82个非空字，97.14%的活跃Tile可以直接使用短列表；room0 RGB-D Full长2480字，实际平均34.36个非空字，99.81%的活跃Tile进入短列表。旧版位表长度随地图N扩大，像素重复扫描大量空字；本轮把这部分浪费移到每Tile一次的构建中。

1%时office2仅178个输入点，却覆盖3220/3225个Tile，room0仅127点也覆盖全部3225个Tile。低点数的补偿Gaussian footprint很大，像素混合和全分辨率图像写入仍存在。因此低Ratio相对上一版的新增收益小于Full，不能继续用点数比例预测渲染时间。这里是单个匹配视角的归因例子，48项工作量记录在TILE_WORKLOADS.csv。

## 正确性证据

- 700组原二进制对照：RGB和radii逐bit一致，包含真实Checkpoint、不同Ratio、全局/局部输入、极薄/大Scale、同深度及非有限输入。
- 额外覆盖全部16场景的896份正式测速输入，逐项验证RGB和radii相同，详见ALL_RENDER_INPUTS_VALIDATION.json。
- 63项生命周期检查：动态K/形状、旧输出独立、预计算RGB/协方差、SH、非默认Stream与跨Stream误复用拒绝。
- 66项新增边界检查：127/128/129成员、1023/1024/1025目录边界、全部不可见输入、同深度排序；直接核对目录、列表前128项及溢出真实count，与原Tile有序成员逐项相同。
- Compute Sanitizer memcheck和initcheck均为0错误，覆盖上述显式边界。Sanitizer子进程的活动不算正式测速。
- 正式输入准备时逐帧比较图像；准备＋渲染测量还逐次验证选点ID与图像相同。模型、点集、补偿属性、相机和分辨率未改。原始空输入CUDA primitive的零输出语义保留，外层背景图特殊处理不在本次改动范围内。

这是冻结推理验证，未修改或声称重新验证训练Backward/Adam；新入口明确只供冻结推理，训练入口继续原路径。已测试图像逐位一致，因此这些图像的PSNR/SSIM/LPIPS不会因本优化改变；未把有限视角测试夸大为全轨迹质量重评。

## 测量协议与干扰控制

RTX3090、CUDA11.8、既有zwj环境；原入口及上一版真实.so均只读加载。全16场景，8个固定轨迹视角，1%/5%/10%/70%、固定1000/5000点与Full。每轮每条件120次、版本顺序轮转并反转，逐次CUDA同步，排除初始化、编译和输入准备。主结论来自每秒采样的两轮quiet确认：纯渲染{n}条、准备＋渲染{np}条。此前较频繁轮询的两轮formal保留作监控开销敏感性对照，另有纯渲染{na}条；独立Profiling不混入墙钟统计。

正式测速前后及全过程保留pmon、GPU时钟/温度/功耗、PID/start_ticks/boot_id。四轮纯渲染和四轮准备＋渲染均未观测到外部活跃GPU负载。正确性验证期间的短暂Xorg活动，以及Sanitizer派生进程活动保留在日志内，均不用于速度结论。每场景每轮Full门限与p90另在CSV中提供，门限未看结果后更改。

进一步按“场景×具体视角×轮次”拆分，{views['full_view_round_cells']}项Full均未退化，最小加速{views['full_min_speedup']:.2f}×。低频监控主测的低Ratio逐视角没有退化；高频监控对照中有{views['formal_low_ratio_view_regressions']}项轻微负收益，最大耗时上升{views['formal_low_ratio_max_latency_increase_pct']:.2f}%。该波动保留，不宣称每一次调用都必然加速。

Ratio用于当前可见名义Anchor候选池的预算，各场景K按自己的候选数量计算；未固定same-K跨场景。不同方法在同一输入内使用相同K和完全相同的Gaussian ID。全16组结果在RASTERIZER_RESULTS.csv，4组真实准备流程在PIPELINE_RESULTS.csv。

固定1000/5000表示请求的局部预算上限；候选池不足时沿用原入口的实际返回数量。例如office1 mono的5000请求，8个视角实际K为1398–4206。报告使用实际K，不把请求值当作成功选中的点数。

## 全16场景Full

'''
    doc+=table(full,[('scene','场景/模式'),('K_max','地图N'),('original_ms','原入口ms'),('bitset_v1_ms','旧Bitset ms'),
        ('warp_v1_ms','旧Warp ms'),('balanced_ms','新版ms'),('speedup_vs_original','相对原入口×'),('speedup_vs_warp_v1','相对旧Warp×')])
    doc+='## 低Ratio代表场景与实际K\n\n'
    doc+=table(selected,[('scene','场景/模式'),('condition','条件'),('K_median','K中位数'),('original_ms','原入口ms'),
        ('bitset_v1_ms','旧Bitset ms'),('balanced_ms','新版ms'),('speedup_vs_bitset_v1','本轮新增×')])
    doc+='## 每次重新准备＋渲染\n\n'
    doc+=table(pipeline,[('scene','场景/模式'),('condition','条件'),('K_median','K中位数'),('original_ms','原入口总ms'),
        ('bitset_v1_ms','旧Bitset总ms'),('balanced_ms','新版总ms'),('balanced_prepare_ms','准备ms'),('balanced_render_ms','渲染ms'),
        ('speedup_vs_bitset_v1','本轮新增×')])
    doc+='## 独立CUDA Event与Kernel归因\n\nCUDA Event计时包含Stream内Host发射空隙，不等于纯Kernel累计；Kineto仅统计cat=kernel，排除GPU annotation，Memcpy/Memset单列。原入口的阻塞DtoH也可能表现为cudaMemcpy相关API等待，不能只凭Synchronize名称统计推断没有同步。\n\n'
    doc+=table(profiles,[('condition','条件'),('variant','版本'),('K','K'),('cuda_event_ms','Event ms'),('gpu_kernel_us','Kernel μs'),
        ('kernels_per_frame','Kernel/帧'),('memcpy_per_frame','Memcpy/帧'),('memset_per_frame','Memset/帧')])
    doc+='''## 候选取舍与剩余瓶颈

两级目录单独路径(mode3)已通过逐位验证，但4场景初筛的Full和10%普遍慢于有序列表路径(mode4)，因此不作为本轮推荐实现。保留其独立开关供消融，不把候选初筛混入正式结果。上一版线性位表(mode2)保留为真实二进制对照，其Full退化已被本轮消除；短列表Warp Shuffle候选来自上轮，未重新宣称有效。没有采用改变浮点累加顺序、近似Alpha或降低分辨率的候选。

剩余热点仍为逐像素Alpha混合、Gaussian→Tile有效覆盖测试和稳定排序。低Ratio下长footprint仍占明显计算，点数减少不等于屏幕工作量同比减少。后续最有信息价值的是按Tile实际密度测量共享属性加载/像素Warp划分，以及为很小的深度序列研究保持稳定次序的排序实现；每项仍需Full不退化及逐位门限。准备＋渲染的新增收益小于纯渲染，因为Selector/Compensator与候选准备在本轮保持不变。

## 完整性与使用边界

实际运行身份另见RUNTIME_IDENTITY.json：Python3.11、PyTorch2.3.1+cu118；/proc/self/maps确认原入口、旧V1及新.so同时加载，CUDA runtime来自zwj的nvidia/cuda_runtime目录，RuntimeGetVersion返回11080。LibTorch同样来自zwj；裸ldd的搜索顺序不等于先import torch后的实际运行路径。独立二进制仅适用于本次服务器环境，其他GPU/环境需重新编译。

'''
    doc+=f'''保护生产源码、配置、二进制、旧研究产物及16个Checkpoint共 **{integrity['files']}项**，研究前后变更 **0项**；候选源码和.so在测量期间不变。未更改生产默认、训练、共享conda、原模型或其他进程，未commit/push。源码、测试与独立二进制位于服务器 `/home/prmi/zwj/experiments/sparse_rasterizer_balanced_20261110`，推荐研究模式为 `SparseRenderer(4)`。本地同名research目录保留全部证据。

RASTERIZER_RESULTS.csv和PIPELINE_RESULTS.csv为主结果；GPU_PROFILE.csv为独立活动测量；MONITOR_SENSITIVITY_RESULTS.csv为额外轮询敏感性复验；FINAL_VERIFICATION.json给出自动验收结果。FULL_RENDER_SPEED.png和LOW_RATIO_RENDER_SPEED.png为性能图，不保存渲染图片。
'''
    (HERE/'FULL_SAFE_LOW_RATIO_CUDA_OPTIMIZATION_REPORT.md').write_text(doc,encoding='utf-8')
    figures(raster,pipeline)
    print(json.dumps(facts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
