"""汇总两轮配对测速和独立Compensator CUDA Graph微测。"""
from pathlib import Path
import csv
import json
import statistics as st

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS=ROOT/'reports'


def read(path):return json.loads(path.read_text())


def main():
    roots=[REPORTS/'paired_speed_20261108',REPORTS/'paired_speed_recheck_20261108']
    graph=read(REPORTS/'component_graph_speed_20261108/summary.json')
    assert graph['passed']
    slugs=read(roots[0]/'COMPLETE.json')['completed']
    assert read(roots[1]/'COMPLETE.json')['completed']==slugs
    combined=[];details=[]
    for slug in slugs:
        summaries=[read(r/slug/'summary.json') for r in roots]
        assert summaries[0]['teacher_fingerprint']==summaries[1]['teacher_fingerprint']
        assert summaries[0]['student_sha256']==summaries[1]['student_sha256']
        assert all(s['selection_ids_identical'] for s in summaries)
        samples=[]
        for run,root in enumerate(roots):
            rows=list(csv.DictReader((root/slug/'prepare_samples.csv').open()))
            assert len(rows)==1800
            for row in rows:
                row.update(run=run)
                samples.append(row)
            grouped={}
            for row in rows:grouped.setdefault((row['condition'],row['repeat'],row['frame']),[]).append(row)
            assert len(grouped)==900 and all(len(g)==2 and len({(r['N'],r['K']) for r in g})==1 for g in grouped.values())
        for condition in ['1000','5000','1pct','10pct','70pct']:
            group=[r for r in samples if r['condition']==condition]
            item=dict(scene=summaries[0]['scene'],N=summaries[0]['map_N'],condition=condition,
                K_median=st.median(int(r['K']) for r in group),K_min=min(int(r['K']) for r in group),
                K_max=max(int(r['K']) for r in group),measured_pairs=len(group)//2)
            for metric in ['prepare_ms','total_ms']:
                times={name:st.median(float(r[metric]) for r in group if r['version']==name) for name in ['teacher','student']}
                reps=[s['prepare'][next(i for i,v in enumerate(s['prepare']) if v['condition']==condition)][metric]['speedup'] for s in summaries]
                six=[v for s in summaries for v in next(v for v in s['prepare'] if v['condition']==condition)[metric]['repeat_speedups']]
                item.update({metric+'_teacher':times['teacher'],metric+'_student':times['student'],
                    metric+'_speedup':times['teacher']/times['student'],metric+'_reduction_pct':100*(1-times['student']/times['teacher']),
                    metric+'_saved':times['teacher']-times['student'],metric+'_run1_speedup':reps[0],metric+'_run2_speedup':reps[1],
                    metric+'_six_repeat_min':min(six),metric+'_six_repeat_max':max(six)})
            combined.append(item)
        details.append(summaries[0])
    with (REPORTS/'COMPENSATOR_DISTILLATION_SPEED.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    lines=['# Compensator蒸馏实际推理速度报告','','## 结论','',
        '比较原64/64四个两层Head的Teacher与32/32四个直接Head的最终学生，两个版本均使用v31冻结CUDA部署后端。结果来自4个代表场景，不代表全部16组或其他GPU。','',
        f'完整准备流程加速范围 {min(r["prepare_ms_speedup"] for r in combined):.3f}–{max(r["prepare_ms_speedup"] for r in combined):.3f}×；准备＋原Rasterizer范围 {min(r["total_ms_speedup"] for r in combined):.3f}–{max(r["total_ms_speedup"] for r in combined):.3f}×。完整准备绝对节省 {min(r["prepare_ms_saved"] for r in combined):.4f}–{max(r["prepare_ms_saved"] for r in combined):.4f}ms，不是毫秒级的普遍节省。','',
        '## 配对条件与计时口径','',
        'RTX 3090，FP32，TF32关闭，CPU线程数2。不改变权重、训练、选择算法和Rasterizer。同一场景Teacher与学生绑定同一Master/Selector/Anchor Feature；每次都更新相机重新计算候选与选点，不复用同帧Mask。图形计划沿用部署后端正常有限形状缓存。','',
        '初始GPU利用率0%，存在658MiB驻留计算进程。逐块核对PID/start_ticks和nvidia-smi pmon，无新增计算进程且驻留进程无SM/显存引擎活动；未杀或调整任何无关进程。观察记录保存在两轮各场景summary.json及组件summary.json。监测为采样，不能排除采样间瞬时系统扰动。','',
        '独立两轮，每轮3次重复；五档请求分别为固定1000/5000点和局部1%/10%/70%。每档先20个跳转相机预热，再计60个视角，每种版本每条件合计360次。Teacher/Student以ABBA次序交替，第二次重复反转条件顺序。实际K相同且选点ID逐对一致。固定预算受候选不足时会截断，表中报告实际K。','',
        '准备耗时从调用prepare到CUDA同步结束；总耗时再包括原渲染入口至CUDA同步。不含磁盘I/O、图像编码、HTTP显示、模型加载、相机构造和离线评价。表中为全部样本中位数，非在线SLAM Mapping测速。由于学生几何预测不同，即使K和ID一致，Rasterizer实际覆盖工作也可能改变；网络本身收益需参考独立组件微测。','',
        '## 实际耗时','',
        '| 场景/模式 | 地图N | 请求 | 实际K中位数 [最小,最大] | 准备Teacher→Student ms | 加速 | 准备＋渲染Teacher→Student ms | 加速 |',
        '|---|---:|---|---|---|---:|---|---:|']
    for r in combined:
        lines.append(f'| {r["scene"]} | {r["N"]:,} | {r["condition"]} | {r["K_median"]:g} [{r["K_min"]},{r["K_max"]}] | {r["prepare_ms_teacher"]:.4f}→{r["prepare_ms_student"]:.4f} | {r["prepare_ms_speedup"]:.3f}× | {r["total_ms_teacher"]:.4f}→{r["total_ms_student"]:.4f} | {r["total_ms_speedup"]:.3f}× |')
    lines+=['','## 独立复测与波动','','| 场景/模式 | 请求 | 准备第一/第二轮加速 | 六次重复范围 | 总耗时第一/第二轮加速 | 六次重复范围 |',
        '|---|---|---|---|---|---|']
    for r in combined:
        lines.append(f'| {r["scene"]} | {r["condition"]} | {r["prepare_ms_run1_speedup"]:.3f}/{r["prepare_ms_run2_speedup"]:.3f} | {r["prepare_ms_six_repeat_min"]:.3f}–{r["prepare_ms_six_repeat_max"]:.3f} | {r["total_ms_run1_speedup"]:.3f}/{r["total_ms_run2_speedup"]:.3f} | {r["total_ms_six_repeat_min"]:.3f}–{r["total_ms_six_repeat_max"]:.3f} |')
    lines+=['','低Ratio的绝对收益较小，不把接近1×的结果解读为稳定显著加速；本次没有锁GPU频率，也没有用并发训练时间推断推理收益。不同场景的原始毫秒数不可脱离N、K、渲染覆盖与CPU调度直接比较。','',
        '## Compensator组件CUDA微测','',
        '固定相同输入、选点和Ratio，单独测Feature查询、顺序Residual累加、MLP、解码和属性发布，排除Selector、视角编码生成和Rasterizer。使用CUDA Event及捕获后的CUDA Graph，复用捕获输出空间，减少Python逐次发射间隙；每点仍执行原内核，不缓存Feature计算结果。每条件3个相机×3次重复×50次Graph replay。','',
        '小地图固定Unrolled16、大地图固定RegisterTile16，对应已有优化内核；此项是组件固定Tile控制实验，不复现完整准备中的全部动态Tile/容量调度。Graph还包含status清零，不能称为纯MLP GEMM耗时。初轮未捕获的tile16 Event数据作为控制单独保存在component_samples.csv，存在Python发射间隙，不作为纯GPU核结论。','',
        '| 场景/模式 | 模式 | Ratio | 实际K中位数 | Teacher ms | Student ms | 加速 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in graph['results']:
        lines.append(f'| {r["scene"]} | {"局部" if r["local"] else "全局"} | {r["ratio"]:g} | {r["K_median"]:g} | {r["teacher"]:.5f} | {r["student"]:.5f} | {r["speedup"]:.3f}× |')
    lines+=['','## 质量与适用限制','',
        '这是牺牲少量拟合精度换速度的学生模型，不是Teacher的数值等价实现。Replica16组的最终学生等权全局平均PSNR损失0.15415dB，局部0.27581dB，原严格质量门限0/352通过。room2单目有一个CUDA对自身PyTorch的图像一致性条件68.336dB未达70dB；具体失败视角CUDA相对普通学生GT PSNR提高约0.00066dB，但失败状态仍保留。','',
        '此次仅增加独立测速脚本与报告，未改模型、训练、生产默认、CUDA内核或Rasterizer，未commit/push。原质量报告中的performance_measured=false描述其训练批次验收时点；本次独立速度证据以本报告和原始计时CSV为准。','',
        '文件：COMPENSATOR_DISTILLATION_SPEED.csv汇总两轮；paired_speed_20261108与paired_speed_recheck_20261108保存所有配对样本及环境；component_graph_speed_20261108保存组件采样。','']
    report='\n'.join(lines)
    (REPORTS/'COMPENSATOR_DISTILLATION_SPEED_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':(HERE/'COMPENSATOR_DISTILLATION_SPEED_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(preparation_speedup_range=[min(r['prepare_ms_speedup'] for r in combined),max(r['prepare_ms_speedup'] for r in combined)],
        total_speedup_range=[min(r['total_ms_speedup'] for r in combined),max(r['total_ms_speedup'] for r in combined)],
        component_speedup_range=[min(r['speedup'] for r in graph['results']),max(r['speedup'] for r in graph['results'])],conditions=len(combined),
        measured_prepare_pairs=sum(r['measured_pairs'] for r in combined)),indent=2))


if __name__=='__main__':main()
