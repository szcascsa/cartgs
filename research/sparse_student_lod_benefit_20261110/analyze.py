"""新G/R下蒸馏的增量收益及相对直接Full的盈亏。"""
from pathlib import Path
import csv,json,statistics as st

HERE=Path(__file__).resolve().parent
REPORTS=HERE/'reports'
RATIOS=[.01,.05,.1,.3,.7]


def read_csv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))


def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    validation=json.loads((REPORTS/'VALIDATION.json').read_text());assert validation['passed']
    rows=read_csv(REPORTS/'samples.csv');assert len(rows)==28512
    checks=read_csv(REPORTS/'correctness.csv');assert len(checks)==792
    quality=read_csv(REPORTS/'quality.csv');assert len(quality)==792
    groups={}
    for r in rows:groups.setdefault((r['scene'],r['scope'],r['round'],r['repeat'],r['frame']),{})[r['condition']]=r
    assert len(groups)==2592 and all(len(v)==11 for v in groups.values())
    for g in groups.values():
        for ratio in RATIOS:assert g[f'teacher_{ratio}']['K']==g[f'student_{ratio}']['K']
    for c in checks:
        assert c['rgb_radii_bitwise']=='True'
        if c['condition'].startswith('student'):
            a=next(x for x in checks if x['scene']==c['scene'] and x['frame']==c['frame'] and x['condition']==c['condition'].replace('student','teacher'))
            assert a['K']==c['K'] and a['ids_sha256']==c['ids_sha256']
    summaries=[];repeated=[];quality_summary=[]
    scenes=sorted({r['scene'] for r in rows})
    for scene in scenes:
        for scope in ['raster','prepare','pipeline']:
            group=[g for k,g in groups.items() if k[0]==scene and k[1]==scope]
            full=st.median(float(g['full']['wall_ms']) for g in group)
            for ratio in RATIOS:
                t=f'teacher_{ratio}';s=f'student_{ratio}'
                tm=st.median(float(g[t]['wall_ms']) for g in group)
                sm=st.median(float(g[s]['wall_ms']) for g in group)
                row=dict(scene=scene,scope=scope,ratio=ratio,N=int(group[0][t]['N']),
                    K_median=st.median(int(g[t]['K']) for g in group),K_min=min(int(g[t]['K']) for g in group),K_max=max(int(g[t]['K']) for g in group),
                    samples=len(group),full_ms=full,teacher_ms=tm,student_ms=sm,
                    student_speedup_vs_teacher=tm/sm,student_speedup_vs_full=full/sm,
                    teacher_speedup_vs_full=full/tm,
                    saved_vs_teacher_paired_ms=st.median(float(g[t]['wall_ms'])-float(g[s]['wall_ms']) for g in group),
                    saved_vs_full_paired_ms=st.median(float(g['full']['wall_ms'])-float(g[s]['wall_ms']) for g in group))
                summaries.append(row)
                for rd in ['0','1']:
                    part=[g for k,g in groups.items() if k[0]==scene and k[1]==scope and k[2]==rd]
                    ft=st.median(float(g['full']['wall_ms']) for g in part)
                    tt=st.median(float(g[t]['wall_ms']) for g in part);ss=st.median(float(g[s]['wall_ms']) for g in part)
                    repeated.append(dict(scene=scene,scope=scope,ratio=ratio,round=rd,full_ms=ft,teacher_ms=tt,student_ms=ss,
                        student_speedup_vs_teacher=tt/ss,student_speedup_vs_full=ft/ss))
        for ratio in RATIOS:
            group=[g for g in quality if g['scene']==scene and g['condition']==f'teacher_{ratio}']
            t=st.mean(float(g['psnr']) for g in group)
            s=st.mean(float(g['psnr']) for g in quality if g['scene']==scene and g['condition']==f'student_{ratio}')
            f=st.mean(float(g['psnr']) for g in quality if g['scene']==scene and g['condition']=='full')
            quality_summary.append(dict(scene=scene,ratio=ratio,views=len(group),full_psnr=f,teacher_psnr=t,student_psnr=s,student_loss_vs_teacher=t-s))
    write_csv(REPORTS/'STUDENT_VS_TEACHER_SPEED.csv',summaries)
    write_csv(REPORTS/'STUDENT_VS_TEACHER_ROUNDS.csv',repeated)
    write_csv(REPORTS/'MATCHED_VIEWS_QUALITY.csv',quality_summary)
    pipeline=[r for r in summaries if r['scope']=='pipeline']
    lines=['# 新渲染器＋蒸馏Compensator配对速度报告','','## 结论','',
        '本轮只替换Compensator执行权重，使用同一新G Renderer、最终R相机、Selector、Master、Anchor Feature、候选池及实际K。比较原Teacher64/64四个两层Head与最终Student32/32四个直接Head；Full采用冻结Master属性直接渲染，不每帧复制整图。','',
        f'4个Replica场景/模式、20个局部Ratio条件中，学生完整流程相对Teacher中位加速范围 {min(r["student_speedup_vs_teacher"] for r in pipeline):.3f}–{max(r["student_speedup_vs_teacher"] for r in pipeline):.3f}×，配对每帧节省 {min(r["saved_vs_teacher_paired_ms"] for r in pipeline):.4f}–{max(r["saved_vs_teacher_paired_ms"] for r in pipeline):.4f}ms。相对直接Full更快的条件，Teacher为 {sum(r["teacher_speedup_vs_full"]>1 for r in pipeline)}/20，Student为 {sum(r["student_speedup_vs_full"]>1 for r in pipeline)}/20；近1×条件仍需看两轮波动，不能泛称所有LoD都比Full快。','',
        'room0的1%档两轮学生相对Teacher约0.997×和1.002×，没有稳定额外加速；该档K很低且大地图候选/Host准备开销仍在。room0的10%档两轮学生均快于直接Full约3.2%–3.3%，Teacher两轮均略慢于Full，这是本轮明确改变盈亏结果的配置。70%虽能节省更多补偿计算，完整学生LoD在四场景仍均慢于直接Full。','',
        '## 实验范围与身份','',
        'office1单目、office2 RGB-D、room0 RGB-D与room2单目。前三组与上一轮Teacher研究相同；第四组替换了fr1_xyz，因为fr1_xyz现有学生的Teacher指纹与上一轮锁定checkpoint不一致，禁止混用，不为测速重训。两轮共28512条计时，792份逐位Renderer验收和792条计时外GT质量数据。','',
        'Student文件为Replica全场景研究中各场景continuous_calibration/best.pt，按完整Teacher/地图/Feature指纹加载，参数3534，Teacher25742。模型路径和SHA、step、fingerprint见MODEL_IDENTITIES.json。','',
        '## 完整推理速度','',
        '包含当前相机CPU求逆及R准备、候选、Selector、Compensator、属性打包、新G渲染与最后同步。Full没有网络推理，仅当前相机与原Master渲染。单位ms，表为同轮配对样本中位数，未混用旧Renderer或不同测试批次。','',
        '| 场景 | Ratio | 实际K中位数 | 直接Full | Teacher LoD | Student LoD | 学生相对Teacher | 学生相对Full | 配对节省Teacher→Student ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in pipeline:
        lines.append(f'| {r["scene"]} | {r["ratio"]:g} | {r["K_median"]:g} | {r["full_ms"]:.4f} | {r["teacher_ms"]:.4f} | {r["student_ms"]:.4f} | {r["student_speedup_vs_teacher"]:.3f}× | {r["student_speedup_vs_full"]:.3f}× | {r["saved_vs_teacher_paired_ms"]:+.4f} |')
    lines+=['','## 两轮独立重复','',
        '| 场景 | Ratio | 第一轮学生/Teacher加速 | 第二轮 | 第一轮学生/Full加速 | 第二轮 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in pipeline:
        a,b=[x for x in repeated if x['scene']==r['scene'] and x['scope']=='pipeline' and x['ratio']==r['ratio']]
        lines.append(f'| {r["scene"]} | {r["ratio"]:g} | {a["student_speedup_vs_teacher"]:.3f}× | {b["student_speedup_vs_teacher"]:.3f}× | {a["student_speedup_vs_full"]:.3f}× | {b["student_speedup_vs_full"]:.3f}× |')
    lines+=['','## 额外收益来自哪里','',
        'prepare仅相机＋完整网络准备，raster仅已准备属性的渲染；两者各自单独测量。各列独立中位数不可简单相加替代完整pipeline。','',
        '| 场景 | Ratio | Teacher准备 ms | Student准备 ms | Teacher渲染 ms | Student渲染 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for r in pipeline:
        p=next(v for v in summaries if v['scene']==r['scene'] and v['ratio']==r['ratio'] and v['scope']=='prepare')
        s=next(v for v in summaries if v['scene']==r['scene'] and v['ratio']==r['ratio'] and v['scope']=='raster')
        lines.append(f'| {r["scene"]} | {r["ratio"]:g} | {p["teacher_ms"]:.4f} | {p["student_ms"]:.4f} | {s["teacher_ms"]:.4f} | {s["student_ms"]:.4f} |')
    lines+=['','Teacher和Student选点身份与K一致，但几何输出不同；Student还可能改变Tile footprint和像素合成工作。完整pipeline加速包含这部分实际渲染差异，不能全部归为MLP计算节省。prepare分项更直接反映网络替换后的准备收益。','',
        '## 同视角质量代价','',
        '本表仅为本轮18个运动/跳转视角的计时外对GT评价，不替代先前32个学生留出视角的质量报告。个别视角可能参与过学生训练或选模，不能称新增留出泛化测试。每个Ratio下Teacher与Student选点相同，Full和LoD质量不保证相等。','',
        '| 场景 | Ratio | Full PSNR | Teacher PSNR | Student PSNR | 学生损失 dB |',
        '|---|---:|---:|---:|---:|---:|']
    for r in quality_summary:
        lines.append(f'| {r["scene"]} | {r["ratio"]:g} | {r["full_psnr"]:.4f} | {r["teacher_psnr"]:.4f} | {r["student_psnr"]:.4f} | {r["student_loss_vs_teacher"]:+.4f} |')
    lines+=['','Replica16组先前正式学生留出结果为全局平均PSNR损失约0.154dB、局部约0.276dB，严格质量等价门限未通过。room2单目此前一个普通学生/CUDA图像一致性条件未达70dB，状态继续保留；本轮只证明新Renderer对该学生的既定CUDA输出逐位一致，不改变已有部署门限。','',
        '## 测量与完整性','',
        'RTX3090、FP32、TF32关闭、CPU线程数2；2轮×6重复×18个连续或跳转视角，第二轮视角逆序，请求顺序逐帧/重复轮转反向。所有请求正式计时前完成18视角预热，Renderer按地图最大容量预编译，未按Ratio调参。模型加载、Graph预编译、GT读取、相机样本构建索引、图片编码/显示、磁盘I/O和在线Tracking/Mapping均不在热态延迟内。','',
        '相机每帧按R重新提交、原inverse错误检查立即执行；候选、Selector、Feature查询、Compensator和渲染每帧重算，没有复用同帧Mask/特征输出。直接Full属性是冻结Master的合法静态输入，不缓存任何相机图像。','',
        f'校验 {validation["correctness_cases"]} 份当前输入的RGB/radii与原A逐位一致；Teacher/Student逐视角K/ID哈希匹配，学生Teacher指纹匹配。保护 {validation["protected_files"]} 项生产/原模型/学生/冻结二进制，SHA变更0项。独立pmon连续采样没有外部GPU活动；未终止或调整驻留进程，未锁GPU频率。采样不能排除更短活动与CPU噪声，不以近1×结果作显著收益承诺。','',
        '新增目录只含实验脚本、报告、数值与性能曲线，未训练、未修改网络/Renderer/生产默认、未commit/push、未保存渲染图片。首次加载与预编译成本沿原Renderer报告保留，短会话不能只看热态延迟。','',
        '第一轮尝试检测到外部PID1456420短暂2% SM活动，整次判为无效，已保留于rejected_attempt01，未混入正式计时；随后负载恢复后完整重跑。拒绝依据是GPU活动，而非耗时高低，未终止外部进程。','',
        '主要文件：STUDENT_VS_TEACHER_SPEED.csv、STUDENT_VS_TEACHER_ROUNDS.csv、MATCHED_VIEWS_QUALITY.csv、samples.csv、correctness.csv、MODEL_IDENTITIES.json、GPU_OBSERVATIONS.json与VALIDATION.json。','',
        '![配对速度](figures/STUDENT_VS_TEACHER_NEW_RENDERER.png)','']
    report='\n'.join(lines)
    (HERE/'STUDENT_NEW_RENDERER_BENEFIT_REPORT.md').write_text(report,encoding='utf-8')
    result=dict(passed=True,samples=len(rows),correctness_cases=len(checks),paired_scope_views=len(groups),conditions=len(pipeline),
        teacher_faster_than_full=sum(r['teacher_speedup_vs_full']>1 for r in pipeline),student_faster_than_full=sum(r['student_speedup_vs_full']>1 for r in pipeline),
        student_speedup_vs_teacher_range=[min(r['student_speedup_vs_teacher'] for r in pipeline),max(r['student_speedup_vs_teacher'] for r in pipeline)],
        student_saved_ms_range=[min(r['saved_vs_teacher_paired_ms'] for r in pipeline),max(r['saved_vs_teacher_paired_ms'] for r in pipeline)])
    (REPORTS/'ANALYSIS_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
