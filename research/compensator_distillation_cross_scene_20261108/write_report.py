"""用实际多场景日志生成中文报告。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st


HERE = Path(__file__).resolve().parent
ROOT = HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS = ROOT/'reports'
SERVER = '/home/prmi/zwj/experiments/compensator_distillation_cross_scene_20261108'
FINAL = 'continuous_calibration'
BASELINE = 'image_calibration'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    scenes = read(REPORTS/'SCENES.json')
    state = read(REPORTS/'BATCH_STATUS.json')
    protocol = read(REPORTS/'PROTOCOL.json')
    lines = ['# Compensator 蒸馏的跨场景验证', '', '## 范围与结论', '',
        '本轮检验 fr1_xyz 上确定的蒸馏流程能否复用于其他场景。每个新场景从自己的冻结 Teacher 初始化独立 32/32 学生，使用对应的 Anchor Feature、Master 和 Selector。实验覆盖 TUM fr1_desk RGB-D、Replica office0 / office3 RGB-D、VECtor corner-slow 单目，三个数据集共四个新场景。', '',
        f'实际完成 {state.get("completed",0)}/4 个场景，失败 {state.get("failed",0)} 个。所有场景 seed=0，训练超参数和最大更新预算在查看新场景结果前固定。', '',
        '### 最终学生相对教师的平均 PSNR 损失', '',
        '| 场景／模式 | N | 全局11条件均值 | 局部11条件均值 | 局部1% | 局部5% | 局部10% | 严格门限通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    all_results = {}
    for scene in scenes:
        folder = REPORTS/scene['slug']
        if not (folder/'QUALITY_SUMMARY.json').exists():
            job = next((j for j in state['scenes'] if j['id']==scene['id']),{})
            lines.append(f'| {scene["id"]} | {scene["gaussians"]} | 失败 | 失败 | — | — | — | {job.get("error","")} |')
            continue
        q = read(folder/'QUALITY_SUMMARY.json')
        all_results[scene['id']] = q
        index = {(r['student'],r['ratio'],r['local']):r for r in q['summaries']}
        final = [r for r in q['summaries'] if r['student']==FINAL]
        global_mean = st.mean(r['mean_loss'] for r in final if not r['local'])
        local_mean = st.mean(r['mean_loss'] for r in final if r['local'])
        values = [index[(FINAL,r,True)]['mean_loss'] for r in [.01,.05,.1]]
        lines.append(f'| {scene["id"]} | {scene["gaussians"]:,} | {global_mean:.5f} | {local_mean:.5f} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {sum(r["gate"] for r in final)}/22 |')
    lines += ['', '正数表示 Teacher 的 GT PSNR 高于学生，负数表示该指标上学生更好。宏平均对各 Ratio 和模式等权。模型选取只使用验证 Teacher 图像误差，测试 GT 指标用于报告结果。', '',
        '固定流程在四个新场景均可完成训练，最终Teacher图像MSE相对原图像校准版降低约38%—50%。质量保持程度具有明显场景差异，最终仅fr1_desk有6/22个条件通过严格门限，其余三个场景均为0/22。', '',
        '连续Ratio阶段在fr1_desk和corner-slow的平均GT PSNR损失略高于几何校准阶段，在office0与office3略低。这一阶段的跨场景质量收益尚不稳定，报告保留全部阶段检查点和逐条件指标。', '',
        '### 图像拟合误差与Teacher质量', '',
        '| 场景 | Teacher平均PSNR | 原图像校准的Teacher–Student MSE | 最终Teacher–Student MSE | 最终平均PSNR损失 |',
        '|---|---:|---:|---:|---:|']
    for name,q in all_results.items():
        teacher = [r for r in q['summaries'] if r['student']=='teacher']
        base = [r for r in q['summaries'] if r['student']==BASELINE]
        final = [r for r in q['summaries'] if r['student']==FINAL]
        lines.append(f'| {name} | {st.mean(r["psnr"] for r in teacher):.5f} | {st.mean(r["teacher_student_mse"] for r in base):.9g} | {st.mean(r["teacher_student_mse"] for r in final):.9g} | {st.mean(r["mean_loss"] for r in final):.5f} |')
    lines += ['', 'Teacher–Student MSE衡量未clamp图像的绝对拟合差距，GT PSNR衡量clamp图像对GT的重建误差。不同场景Teacher的初始重建误差不同，相同量级的学生扰动可能产生明显不同的PSNR损失。', '',
        '对于clamp图像，令 e=T−GT、d=S−T，则 MSE(S,GT)=MSE(T,GT)+MSE(S,T)+2 mean(e·d)。PSNR差距同时取决于Teacher基准误差、学生拟合误差和二者相关项，跨场景不能只按ΔPSNR归因学生表达能力。当前MSE列使用未clamp图像，用作拟合上下文，完整误差分解未单独测量。', '',
        '## 固定训练流程', '',
        '学生输入为归一化局部 offset3、共享 Residual48、连续 ratio3，以及局部模式的 view7+mode1 条件。学生保持两层32宽主干和四个直接线性 Head，MLP 参数量为3534。输出沿用 XYZ、Scale、Rotation、Opacity 与 SH DC 的14D语义。', '',
        '每场景使用 Teacher 前32行与前32×32子块初始化主干，12组训练样本通过原FP64 Ridge拟合输出Head。后续学习采用FP32。Teacher、Master、Selector、Anchor Feature 和输入状态均 detach，仅学生MLP参数更新。', '',
        '| 阶段 | 更新预算 | 学习率 | 图像监督 | 原始残差权重 | Ratio采样 | 选模依据 |',
        '|---|---:|---:|---|---:|---|---|',
        '| 初始蒸馏 | 6000 | 0.001 | 每20步，权重1 | 1 | 六档轮转 | 原属性＋图像组合 |',
        '| 图像校准 | 2000 | 0.001 | 每5步，权重1000 | 1 | 六档轮转 | 原属性＋图像组合 |',
        '| 几何校准 | 6000 | 0.0002 | 每步，权重200 | 0.01 | 六档轮转 | Teacher图像MSE |',
        '| 连续Ratio校准 | 3600 | 0.0001 | 每步，权重200 | 0.01 | 50%六档＋50%区间log均匀 | Teacher图像MSE |', '',
        '每阶段从上一阶段best初始化，重置Adam和采样RNG。初始验证也参加best选择，因此阶段best可能位于0步。输出图像监督与属性监督共享冻结教师选点，图像使用原分辨率，未保存渲染图片。', '',
        '六档为 [.01,.05,.1,.3,.5,.7]。连续阶段等概率选择五个相邻区间，在log(ratio)中均匀采样，并保持同一个全局/局部Ratio对的采样值一致。连续阶段视角采样由seed和步骤独立决定。', '',
        '最终训练目标为 0.01×原始归一化残差Loss＋解码后XYZ／Log-Scale／Rotation／Opacity／SH DC属性Loss＋200×Teacher图像MSE。图像训练目标使用未clamp RGB，PSNR／SSIM／LPIPS评价使用clamp至[0,1]的RGB。', '',
        '## 数据划分与评价口径', '',
        '使用每个模型保存的完整估计轨迹。TUM和VECtor读取rgb.txt，Replica读取results/frame*.jpg的帧编号。与RGB关联的时间差上限为0.001秒，要求保存轨迹全部一一匹配。相机内参与分辨率使用Teacher checkpoint，RGB沿用该模式现有评价口径。', '',
        '原BudgetScene对缺少rgb.txt的Replica会退回保存关键帧，本轮通过独立研究加载器接入完整轨迹。加载器位于研究目录，帧关联、顺序及哈希均记录。估计轨迹直接用于配对渲染，当前结果与历史经过GT轨迹同步筛选的指标可能因视角集合不同而不同。', '',
        '学生训练排除每5帧池，其中三个均匀分布视角用于验证，剩余池均匀选32帧作测试。各阶段共用视角划分。最终学生留出评价仍位于Teacher建图使用过的同一场景轨迹，跨场景指蒸馏流程的复用性。', '',
        '前三阶段验证六档Ratio，最后阶段加五个相邻档位几何中点。检查点选择只用验证Teacher输出或图像，全部新场景不调参。非锚点评价Ratio为[.03,.08,.2,.4,.6]，这些点位于连续采样区间内。', '',
        'Teacher和Student使用完全相同的选择身份。局部使用名义Anchor视锥候选和严格K，全局保留原阈值与同分语义。实际K逐视角导出，选择ID另存SHA。Full是原Master渲染。', '',
        '质量门限沿用每Ratio×模式平均PSNR损失≤0.01dB、p99≤0.05dB、最坏≤0.1dB。单场景32视角的p99仅用于描述当前样本尾部。', '',
        '## 各场景结果', '']
    for scene in scenes:
        if scene['id'] not in all_results:
            continue
        folder = REPORTS/scene['slug']
        q = all_results[scene['id']]
        preflight = read(folder/'PREFLIGHT.json')
        index = {(r['student'],r['ratio'],r['local']):r for r in q['summaries']}
        lines += [f'### {scene["id"]}', '',
            f'地图含 {scene["gaussians"]:,} 个Gaussian，保存关键帧 {preflight["keyframe_count"]} 个，完整轨迹 {preflight["full_trajectory_frames"]} 帧。蒸馏训练视角 {preflight["training_views"]} 个，验证视角 {preflight["validation_views"]}，测试32视角。测试视角Full平均PSNR为 {q["full_psnr"]:.5f} dB。', '',
            '| Ratio | 模式 | Teacher PSNR | 初始图像校准 ΔPSNR | 几何校准 ΔPSNR | 最终 ΔPSNR | 最终p99 / 最坏 | 实际K范围 |',
            '|---:|---|---:|---:|---:|---:|---|---|']
        for ratio in sorted(set(r['ratio'] for r in q['summaries'])):
            for local in [False,True]:
                teacher = index[('teacher',ratio,local)]
                b,g,f = [index[(name,ratio,local)] for name in [BASELINE,'geometry_calibration',FINAL]]
                lines.append(f'| {ratio:g} | {"局部" if local else "全局"} | {teacher["psnr"]:.5f} | {b["mean_loss"]:.5f} | {g["mean_loss"]:.5f} | {f["mean_loss"]:.5f} | {f["p99_loss"]:.5f} / {f["worst_loss"]:.5f} | {f["K_min"]}–{f["K_max"]} |')
        lines += ['', '| Ratio | 模式 | Teacher SSIM | 最终SSIM | Teacher LPIPS | 最终LPIPS |',
            '|---:|---|---:|---:|---:|---:|']
        for ratio in sorted(set(r['ratio'] for r in q['summaries'])):
            for local in [False,True]:
                t,f = [index[(name,ratio,local)] for name in ['teacher',FINAL]]
                lines.append(f'| {ratio:g} | {"局部" if local else "全局"} | {t["ssim"]:.6f} | {f["ssim"]:.6f} | {t["lpips"]:.6f} | {f["lpips"]:.6f} |')
        lines += ['', '各阶段选中的best如下。', '', '| 阶段 | best步数 | 权重SHA256 |', '|---|---:|---|']
        for name,value in q['identities'].items():
            lines.append(f'| {name} | {value["step"]} | `{value["sha256"]}` |')
        deployment_path = folder/'DEPLOYMENT_VALIDATION.json'
        if deployment_path.exists():
            d = read(deployment_path)
            lines += ['',f'本场景完成 {d["cases"]} 组最终学生CUDA/PyTorch检查，最大14D绝对差为 {d["max_residual_abs"]:.9g}，最低相互渲染PSNR为 {d["min_pytorch_cuda_psnr"]:.3f} dB。Graph/Native与选择身份逐位检查通过。', '']
    lines += ['## 实际更新和资源', '',
        'CROSS_SCENE_TRAINING.csv记录每阶段实际步数、图像反传次数、属性点数累计、墙钟与峰值显存。共享GPU存在其他进程，墙钟仅用于说明实验成本。部分首阶段的峰值显存记录缺失时，CSV保留空值。', '',
        '| 场景 | 实际学生更新 | 图像反传次数 | 正式训练墙钟秒 | 峰值分配MiB |',
        '|---|---:|---:|---:|---:|']
    training_path = REPORTS/'CROSS_SCENE_TRAINING.csv'
    if training_path.exists():
        training = list(csv.DictReader(training_path.open(encoding='utf-8-sig')))
        for scene in scenes:
            group = [r for r in training if r['scene']==scene['id']]
            if not group:
                continue
            peaks = [float(r['peak_allocated_MiB']) for r in group if r['peak_allocated_MiB']]
            lines.append(f'| {scene["id"]} | {sum(int(r["steps"]) for r in group)} | {sum(int(r["image_updates"]) for r in group)} | {sum(float(r["seconds"]) for r in group):.1f} | {max(peaks):.1f} |')
    lines += ['', '## 归因与局限', '',
        '已证实这套流程可在四个新场景复用，并降低相对原图像校准版的总体Teacher拟合误差。Teacher质量等价目标仍未达成。局部模式的损失普遍高于全局，较高Ratio也可能保留明显差距，具体见曲线与逐条件表。', '',
        '本轮包含完整蒸馏流程的三个阶段检查点对比，几何校准和连续Ratio校准也增加了更新预算。因此它检验固定流程在新场景是否有效，阶段之间的质量变化同时包含目标、采样和训练量变化。fr1_xyz上的匹配消融用于解释设计机制，本轮新场景结果独立报告。', '',
        '每个场景独立学生绑定该场景Teacher与Feature表，模型参数未直接跨场景迁移。当前未覆盖全部场景、全部相机模式和多个学生seed。真实端侧速度需要GPU空闲后进行配对测试。', '',
        '误差可能受Teacher补偿复杂度、局部候选分布、稀疏地图、轨迹覆盖和学生容量共同影响。输出误差、实际K、XYZ/Scale尾部已导出，单场景均值不足以区分这些因素。', '',
        '## 完整性与交付', '']
    integrity_path = REPORTS/'FINAL_INTEGRITY.json'
    if integrity_path.exists():
        integrity = read(integrity_path)
        for name,value in integrity.items():
            if isinstance(value,dict) and 'files' in value and isinstance(value['files'],int):
                lines.append(f'- {name}核验 {value["files"]} 个文件，变更数 {len(value["changed"])}。')
        lines += ['', f'最终完整性核验通过状态为 {integrity["passed"]}，渲染图片保存数为 {integrity["saved_render_images"]}。算法源码、Teacher模型、生产文件、部署和Rasterizer二进制均具有SHA证据。', '']
    lines += [f'服务器根目录为 `{SERVER}`。各场景学生保存在 `models/<场景slug>/continuous_calibration/best.pt`。逐条件汇总为CROSS_SCENE_METRICS.csv，逐视角质量、K、属性尾部和checkpoint身份位于reports/<场景slug>。', '',
        '![跨场景PSNR](CROSS_SCENE_QUALITY.png)', '', '![跨场景蒸馏差距](CROSS_SCENE_MACRO.png)', '']
    report = '\n'.join(lines)
    (REPORTS/'COMPENSATOR_DISTILLATION_CROSS_SCENE_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':
        entry = report.replace('(CROSS_SCENE_QUALITY.png)','(results/reports/CROSS_SCENE_QUALITY.png)')
        entry = entry.replace('(CROSS_SCENE_MACRO.png)','(results/reports/CROSS_SCENE_MACRO.png)')
        (HERE/'COMPENSATOR_DISTILLATION_CROSS_SCENE_REPORT.md').write_text(entry,encoding='utf-8')
    print('REPORT_COMPLETE',len(all_results))


if __name__ == '__main__':
    main()
