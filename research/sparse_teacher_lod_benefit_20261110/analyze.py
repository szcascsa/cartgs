"""归因网络LoD是否快于同一新渲染器的直接Master Full。"""
from pathlib import Path
import csv,json,statistics as st

HERE=Path(__file__).resolve().parent
REPORTS=HERE/'reports'
REFERENCE=HERE.parent/'sparse_rasterizer_sixhour_20261110_0053'
if not REFERENCE.exists():REFERENCE=Path('/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053')
CONDITIONS=['full_direct','full_copy','local0.01','local0.05','local0.1','local0.3','local0.7']


def read_csv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))


def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    validation=json.loads((REPORTS/'VALIDATION.json').read_text());assert validation['passed'] and not validation['student']
    rows=read_csv(REPORTS/'samples.csv');assert len(rows)==12096
    groups={}
    for row in rows:
        key=(row['scene'],row['scope'],row['round'],row['repeat'],row['frame'])
        groups.setdefault(key,{})[row['condition']]=row
    assert len(groups)==1728 and all(set(g)==set(CONDITIONS) for g in groups.values())
    summaries=[];rounds=[]
    for scene in sorted({r['scene'] for r in rows}):
        for scope in ['raster','pipeline']:
            g={cond:[r for r in rows if r['scene']==scene and r['scope']==scope and r['condition']==cond] for cond in CONDITIONS}
            full_median=st.median(float(r['wall_ms']) for r in g['full_direct'])
            for cond in CONDITIONS:
                group=g[cond];times=[float(r['wall_ms']) for r in group]
                pairs=[(float(v['full_direct']['wall_ms']),float(v[cond]['wall_ms']))
                    for k,v in groups.items() if k[0]==scene and k[1]==scope]
                item=dict(scene=scene,scope=scope,condition=cond,N=int(group[0]['N']),samples=len(group),
                    K_median=st.median(int(r['K']) for r in group),K_min=min(int(r['K']) for r in group),K_max=max(int(r['K']) for r in group),
                    median_ms=st.median(times),full_direct_median_ms=full_median,
                    speedup_vs_direct_full=full_median/st.median(times),
                    paired_saved_median_ms=st.median(a-b for a,b in pairs),
                    paired_faster_views=sum(a>b for a,b in pairs),
                    round0_median_ms=st.median(float(r['wall_ms']) for r in group if r['round']=='0'),
                    round1_median_ms=st.median(float(r['wall_ms']) for r in group if r['round']=='1'))
                summaries.append(item)
                for index in ['0','1']:
                    current=[r for r in group if r['round']==index]
                    base=[r for r in g['full_direct'] if r['round']==index]
                    rounds.append(dict(scene=scene,scope=scope,condition=cond,round=index,
                        median_ms=st.median(float(r['wall_ms']) for r in current),
                        full_median_ms=st.median(float(r['wall_ms']) for r in base),
                        speedup=st.median(float(r['wall_ms']) for r in base)/st.median(float(r['wall_ms']) for r in current)))
    write_csv(REPORTS/'TEACHER_LOD_SPEED_SUMMARY.csv',summaries)
    write_csv(REPORTS/'TEACHER_LOD_SPEED_ROUNDS.csv',rounds)
    historical=read_csv(REFERENCE/'FINAL_ASYNC_PIPELINE_RESULTS.csv')
    previous=[]
    for row in historical:
        if row['variant']!='R' or not row['condition'].startswith('local'):continue
        full=next(r for r in historical if r['scene']==row['scene'] and r['condition']=='full' and r['variant']=='R')
        previous.append(dict(scene=row['scene'],condition=row['condition'],N=row['N'],K_median=row['K_median'],
            full_copy_ms=float(full['median_ms']),lod_ms=float(row['median_ms']),
            saved_vs_full_copy_ms=float(full['median_ms'])-float(row['median_ms']),
            speedup_vs_full_copy=float(full['median_ms'])/float(row['median_ms'])))
    assert len({r['scene'] for r in previous})==24
    write_csv(REPORTS/'EXISTING_24_SCENES_VS_COPY_FULL.csv',previous)
    lines=['# 新渲染器＋未蒸馏网络：减少Gaussian是否带来实际提速','','## 结论','',
        '答案是有条件的：同一已验收新Renderer下，LoD减少点数会明显降低纯渲染时间；计入原64/64 Compensator、Selector和相机准备后，完整流程只在部分场景/低Ratio有净收益。不能根据新Renderer相对旧Renderer的加速，直接认定选择/补偿网络本身带来了相同收益。','',
        '新增复测4场景，2轮×6次重复×18个运动/跳转视角，7种请求，纯渲染与完整流程共12096条计时样本；504份输入与原A图像/radii逐位一致。未蒸馏Compensator参数25742，所有实例student=None；原Selector、Master、Feature、Ratio和补偿属性保持不变。','',
        '## 必须区分两个Full口径','',
        '- full_direct：冻结Master的最终物理属性包在加载时准备一次，之后每帧只更新相机并渲染。没有Selector、Compensator，也没有重复复制完整Master。此处保留静态地图数据是合法推理准备，不是缓存相机/图像或动态网络输出。','- full_copy：原六小时报告的Full产品接口，每帧runtime.full()复制属性包及ID，再渲染。与历史报告保持可追溯，但不是最轻量的Full对照。','',
        '本轮统一使用G Renderer和最终R相机准备，Full/LoD使用同一GPU、checkpoint、18个视角、原始分辨率及背景。局部预算N为可见名义Anchor内Gaussian数，K受资格数限制；不是全图N乘Ratio，也不强行same-K跨场景。','',
        '## 本轮完整流程结果','',
        '包含每帧CPU位姿求逆、相机R、候选/Selector/原Compensator/打包及G渲染，最后同步。下面均为中位数，ms；不含数据读取、图像编码、Viewer显示、Tracking/Mapping或模型加载。','',
        '| 场景 | 地图N | 直接Full | 每帧复制Full | LoD1% | LoD5% | LoD10% | LoD30% | LoD70% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    scenes=sorted({r['scene'] for r in summaries})
    def find(scene,scope,condition):return next(r for r in summaries if r['scene']==scene and r['scope']==scope and r['condition']==condition)
    for scene in scenes:
        values=[find(scene,'pipeline',c) for c in CONDITIONS]
        lines.append('| '+scene+' | '+f'{values[0]["N"]:,}'+' | '+' | '.join(f'{r["median_ms"]:.4f}' for r in values)+' |')
    lines+=['','## 净节省与实际K','',
        '净节省为同轮、同重复、同视角的直接Full耗时减去LoD耗时，再取配对中位数；正数表示LoD更快。逐条件汇总的中位数相减可能与配对节省略有区别。','',
        '| 场景 | Ratio | 实际K中位数 [最小,最大] | 相对直接Full加速 | 配对节省 ms | 第一轮加速 | 第二轮加速 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for scene in scenes:
        for cond in CONDITIONS[2:]:
            row=find(scene,'pipeline',cond)
            reps=[r['speedup'] for r in rounds if r['scene']==scene and r['scope']=='pipeline' and r['condition']==cond]
            lines.append(f'| {scene} | {cond[5:]} | {row["K_median"]:g} [{row["K_min"]},{row["K_max"]}] | {row["speedup_vs_direct_full"]:.3f}× | {row["paired_saved_median_ms"]:+.4f} | {reps[0]:.3f}× | {reps[1]:.3f}× |')
    lines+=['','## 纯Renderer收益','',
        '同一新G Renderer，属性包和相机在计时外准备；每次render仍重新读取Gaussian并计算预处理、Tile任务和合成，没有复用图像内容。Full与LoD都使用新Renderer，避免将Renderer升级收益误记成减少点数收益。','',
        '| 场景 | Full ms | 1% ms | 5% ms | 10% ms | 30% ms | 70% ms | 1%相对Full加速 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scene in scenes:
        group=[find(scene,'raster',c) for c in ['full_direct']+CONDITIONS[2:]]
        lines.append('| '+scene+' | '+' | '.join(f'{r["median_ms"]:.4f}' for r in group)+f' | {group[1]["speedup_vs_direct_full"]:.3f}× |')
    lines+=['','## 六小时报告24组的既有证据','',
        '既有两轮覆盖24组；当前只读汇总，没有因模型或代码未变而重复全部测试。它采用full_copy口径。不要将下面结论直接写成“相对充分优化的直接Full全部场景都更快”。','',
        '| Ratio | 中位完整耗时快于copy Full的场景数 | 配对场景中位净节省 ms | 场景节省范围 ms |',
        '|---|---:|---:|---:|']
    for cond in ['local0.01','local0.05','local0.1','local0.3','local0.7']:
        group=[r for r in previous if r['condition']==cond]
        vals=[r['saved_vs_full_copy_ms'] for r in group]
        lines.append(f'| {cond[5:]} | {sum(v>0 for v in vals)}/24 | {st.median(vals):+.4f} | {min(vals):+.4f}–{max(vals):+.4f} |')
    lines+=['','第一行24/24是场景合并中位数结论，不等于所有单视角均更快；原报告已有负收益单视角，继续保留。新增直接Full对照发现，copy Full每帧还含约0.06ms发布/复制与Host调度成本，可能改变小场景的LoD盈亏结论。','',
        '## 解释与研究边界','',
        '1. 新Renderer确实使低点数的渲染成本下降；在本轮4场景中，1%纯渲染约0.076–0.114ms，Full约0.165–0.410ms。实际毫秒值以表格为准，不能用旧Renderer约0.8ms的固定开销继续解释本轮结果。','2. 选择、补偿和候选准备仍需时间。较大地图/较低Ratio能用渲染节省覆盖这些开销；小地图或高Ratio则往往覆盖不了。未蒸馏条件下，本轮fr1_xyz所有被测LoD均慢于直接Full。','3. 全图N、局部候选N与实际K是不同数量。补偿还会改变Scale/Rotation/XYZ及屏幕覆盖，K的压缩率不等于Tile/像素计算量压缩率。六小时报告的相同ID补偿对照已经给出低Ratio footprint放大的证据，本轮不修改补偿以制造速度收益。','4. Full与各Ratio的渲染质量不同。本轮只验证资源速度，不重新训练或宣称质量相同；新Renderer对各自输入的RGB/radii与原A逐位一致，不能据此宣称LoD图像等于Full。','',
        '结论应限定为“在适合的地图规模和预算下，网络减少点数可以提高完整推理速度”；目前证据不支持所有场景统一开启LoD总能提速，也不支持固定点数越少越快或按压缩率线性推算。','',
        '## 正确性、负载及完整性','',
        f'新测{validation["correctness_cases"]}份输入图像与radii逐位通过，R相机与原始相机对应选点身份一致；{validation["protected_files"]}项原文件/模型/生产/冻结二进制SHA前后无变化。GPU连续pmon采样未检测到其他进程SM/显存引擎活动，驻留进程未终止或修改。1Hz左右采样不能排除更短外部活动或CPU噪声，GPU频率未锁定。','',
        '所有代码位于独立研究目录，原六小时目录只读加载；直接核对冻结.so和R相机源码哈希，不触发JIT。新Renderer始终以全地图容量预编译，不针对不同Ratio改算法。启动/模型加载/预编译不计入热态延迟，成本沿原报告单独披露。没有训练、生产切换、commit/push或渲染图片保存。','',
        '原始证据：reports/samples.csv、correctness.csv、MODEL_IDENTITIES.json、GPU_OBSERVATIONS.json、PROTECTED_BEFORE/AFTER.json、VALIDATION.json；汇总为TEACHER_LOD_SPEED_SUMMARY.csv和TEACHER_LOD_SPEED_ROUNDS.csv。旧24组结果单列EXISTING_24_SCENES_VS_COPY_FULL.csv。','',
        '![新Renderer下的Full与LoD](figures/TEACHER_LOD_VS_FULL.png)','']
    report='\n'.join(lines)
    (HERE/'TEACHER_LOD_RENDERING_BENEFIT_REPORT.md').write_text(report,encoding='utf-8')
    print('REPORT_DONE',len(summaries),len(previous))


if __name__=='__main__':main()
