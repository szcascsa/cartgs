"""汇总Replica八场景两种相机模式的统一蒸馏结果。"""
from pathlib import Path
import csv
import json
import statistics as st


HERE = Path(__file__).resolve().parent
ROOT = HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS = ROOT/'reports'
FINAL = 'continuous_calibration'
INITIAL = 'image_calibration'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    scenes = read(REPORTS/'SCENES.json')
    state = read(REPORTS/'BATCH_STATUS.json')
    completed = sum(j['status']=='complete' for j in state['scenes'])
    failed = sum(j['status']=='failed' for j in state['scenes'])
    data = {s['id']:read(REPORTS/s['slug']/'QUALITY_SUMMARY.json') for s in scenes
        if (REPORTS/s['slug']/'QUALITY_SUMMARY.json').exists()}
    trained = sum(j.get('training_complete',False) or j['status']=='complete' for j in state['scenes'])
    lines = ['# Replica 全场景 Compensator 蒸馏报告','','## 实验范围与完成状态','',
        f'Replica office0–4、room0–2，单目和RGB-D共16组。训练完成 {trained}/16，质量评价完成 {len(data)}/16，全流程部署验收通过 {completed}/16，失败 {failed}。相机模式沿用用户此前确认的Replica单目＋RGB-D。',
        '', '所有场景使用相同32/32学生、四个直接输出Head、3534个MLP参数、seed=0和四阶段最大17600步预算，每个学生绑定自身冻结Teacher、Master、Selector及Anchor Feature。',
        '', 'office0 RGB-D和office3 RGB-D复用上一轮结果，复用前核对同一个Teacher、场景输入、四阶段配置、更新步数、学生SHA和部署验收。其余14组本轮训练，新增计划246400步。复用模型复制到本轮独立目录，原文件列入保护清单。',
        '', '## 最终学生质量差距','',
        '下表的PSNR损失为同一GT上的Teacher PSNR减去Student PSNR，正数表示学生退化。全局和局部均值分别对11个Ratio等权。',
        '', '| 场景 | 相机 | N | 全局平均损失 | 局部平均损失 | 局部1% | 局部5% | 局部10% | 严格门限通过 | 复用 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for scene in scenes:
        if scene['id'] not in data:
            job = next((j for j in state['scenes'] if j['id']==scene['id']),{})
            lines.append(f'| {scene["scene"]} | {scene["mode"]} | {scene["gaussians"]} | {job.get("status","pending")} | — | — | — | — | — | 否 |')
            continue
        q = data[scene['id']]
        final = [r for r in q['summaries'] if r['student']==FINAL]
        index = {(r['ratio'],r['local']):r for r in final}
        gm = st.mean(r['mean_loss'] for r in final if not r['local'])
        lm = st.mean(r['mean_loss'] for r in final if r['local'])
        values = [index[(r,True)]['mean_loss'] for r in [.01,.05,.1]]
        lines.append(f'| {scene["scene"]} | {scene["mode"]} | {scene["gaussians"]:,} | {gm:.5f} | {lm:.5f} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {sum(r["gate"] for r in final)}/22 | {"是" if scene["reused"] else "否"} |')
    finals = [r for q in data.values() for r in q['summaries'] if r['student']==FINAL]
    if finals:
        lines += ['',f'最终学生共有 {sum(r["gate"] for r in finals)}/{len(finals)} 个场景/模式/Ratio条件通过全部严格质量门限。等权全局平均PSNR损失 {st.mean(r["mean_loss"] for r in finals if not r["local"]):.5f}dB，局部平均损失 {st.mean(r["mean_loss"] for r in finals if r["local"]):.5f}dB。当前32/32学生不能视为Teacher质量等价替代，部署一致性检查与Teacher蒸馏质量门限是两项独立验证。','']
    lines += ['', '## 训练协议','',
        '| 阶段 | 最大更新数 | 学习率 | 图像监督 | 原始残差权重 | Ratio采样 |',
        '|---|---:|---:|---|---:|---|',
        '| 初始蒸馏 | 6000 | 0.001 | 每20步，权重1 | 1 | 六档轮转 |',
        '| 图像校准 | 2000 | 0.001 | 每5步，权重1000 | 1 | 六档轮转 |',
        '| 几何校准 | 6000 | 0.0002 | 每步，权重200 | 0.01 | 六档轮转 |',
        '| 连续Ratio校准 | 3600 | 0.0001 | 每步，权重200 | 0.01 | 50%六档＋50%区间log均匀 |','',
        '六档Ratio为[.01,.05,.1,.3,.5,.7]。非锚点评价Ratio为[.03,.08,.2,.4,.6]，位于连续采样区间内。各阶段从上阶段best初始化，重置Adam与采样RNG。初始验证参与best选择，因此阶段best可能为第0步。','',
        '解码后的XYZ、Log-Scale、Rotation、Opacity、SH DC属性监督保持原公式。Teacher、Master、Anchor Feature、Selector及输入状态detach，仅Student更新。图像目标使用同一个选点集合、相机和原分辨率下的Teacher RGB。','',
        '前三阶段验证六档，最后阶段加入五个相邻Ratio几何中点。检查点选择使用属性组合或Teacher图像MSE，GT只用于测试。所有新场景不调超参数。','',
        '## 完整轨迹与评价','',
        '从保存的CameraTrajectory_TUM.txt读取完整估计轨迹，以Replica results/frame*.jpg编号关联RGB，阈值0.001秒，要求全部保存轨迹帧一一匹配。蒸馏训练使用每5帧池以外的视角，池内3个均匀分布视角用于选模，剩余池固定32帧作测试。Teacher和地图来自同一条轨迹，留出专指学生蒸馏视角。','',
        '全局、局部分别比较Teacher和三个阶段学生，逐视角核对相同选点ID与实际K。局部沿用名义Anchor候选和严格K，全局保留原阈值与同分语义。记录PSNR、SSIM、LPIPS、Teacher–Student图像MSE、XYZ/Scale长尾、实际预算与模型哈希。','',
        '质量门限为每Ratio每模式平均PSNR损失≤0.01dB、p99≤0.05dB、最坏≤0.1dB。32视角的p99仅描述当前样本尾部。','',
        '## 阶段对比与Teacher误差背景','',
        '| 场景 | 相机 | Teacher平均PSNR | 原图像校准损失 | 几何校准损失 | 最终损失 | 最终Teacher–Student MSE |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scene in scenes:
        if scene['id'] not in data:
            continue
        q = data[scene['id']]
        groups = {name:[r for r in q['summaries'] if r['student']==name]
            for name in ['teacher',INITIAL,'geometry_calibration',FINAL]}
        values = [st.mean(r['mean_loss'] for r in groups[name]) for name in [INITIAL,'geometry_calibration',FINAL]]
        lines.append(f'| {scene["scene"]} | {scene["mode"]} | {st.mean(r["psnr"] for r in groups["teacher"]):.5f} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} | {st.mean(r["teacher_student_mse"] for r in groups[FINAL]):.9g} |')
    lines += ['', 'Teacher基准质量不同，相似的学生图像扰动会对应不同的GT PSNR损失。Teacher–Student MSE和相对GT质量同时报告。阶段之间同时改变目标、采样和训练量，阶段差异作为流程结果，单项机制因果仍依赖匹配消融。','',
        '## 每场景逐Ratio结果','']
    for scene in scenes:
        if scene['id'] not in data:
            continue
        q = data[scene['id']]
        folder = REPORTS/scene['slug']
        preflight = read(folder/'PREFLIGHT.json')
        index = {(r['student'],r['ratio'],r['local']):r for r in q['summaries']}
        lines += [f'### {scene["scene"]} / {scene["mode"]}','',
            f'Gaussian {scene["gaussians"]:,}，Anchor {q["anchor_count"]:,}，完整轨迹 {preflight["full_trajectory_frames"]} 帧，测试Full PSNR {q["full_psnr"]:.5f}dB。测试和验证帧列表见对应PREFLIGHT.json。','',
            '| Ratio | 模式 | Teacher PSNR | Student PSNR | 平均损失 | p99 / 最坏 | Teacher / Student SSIM | Teacher / Student LPIPS | K范围 |',
            '|---:|---|---:|---:|---:|---|---|---|---|']
        for ratio in sorted(set(r['ratio'] for r in q['summaries'])):
            for local in [False,True]:
                t,f = [index[(name,ratio,local)] for name in ['teacher',FINAL]]
                lines.append(f'| {ratio:g} | {"局部" if local else "全局"} | {t["psnr"]:.5f} | {f["psnr"]:.5f} | {f["mean_loss"]:.5f} | {f["p99_loss"]:.5f} / {f["worst_loss"]:.5f} | {t["ssim"]:.6f} / {f["ssim"]:.6f} | {t["lpips"]:.6f} / {f["lpips"]:.6f} | {f["K_min"]}–{f["K_max"]} |')
        lines += ['', '| 阶段 | best步数 | checkpoint SHA256 |','|---|---:|---|']
        for stage,value in q['identities'].items():
            lines.append(f'| {stage} | {value["step"]} | `{value["sha256"]}` |')
        validation = folder/'DEPLOYMENT_VALIDATION.json'
        if validation.exists():
            d = read(validation)
            lines += ['',f'部署检查 {d["cases"]} 组通过，CUDA/PyTorch学生输出最大绝对差 {d["max_residual_abs"]:.9g}，最低相互渲染PSNR {d["min_pytorch_cuda_psnr"]:.3f}dB。Graph/Native及选择身份逐位检查见GRAPH_PARITY.json。','']
        diagnostic = folder/'DEPLOYMENT_DIAGNOSTIC.json'
        if diagnostic.exists():
            d = read(diagnostic)
            lines += ['',f'独立部署诊断覆盖 {d["cases"]} 个条件，原70dB门限未通过 {len(d["failed_cases"])} 个；Graph/Native和选择身份逐位检查通过状态 {d["graph_bitwise_passed"]}。该诊断不替代原失败验收，也不放宽门限。','']
            for row in d['failed_cases']:
                lines += [f'帧 {row["frame"]}，Ratio={row["ratio"]}，局部={row["local"]}，K={row["K"]}：普通学生与融合部署相互PSNR {row["pytorch_cuda_psnr"]:.5f}dB；残差最大绝对差 {row["residual_max_abs"]:.9g}，通过原残差张量容差。只替换网络残差得到 {row["raw_only_psnr"]:.5f}dB，只比较属性解码得到 {row["decode_only_psnr"]:.5f}dB。单独将Scale改回普通学生值后为 {row["field_replacement_psnr"]["scales"]:.5f}dB，将Rotation改回后为 {row["field_replacement_psnr"]["rotations"]:.5f}dB，误差主要来自网络FP32输出微扰作用于几何属性。具体Rasterizer分支或排序敏感性尚未单独隔离，不能称为已证明的原因。','',
                    '此项保留失败状态和原日志，未重训、未改模型、未改Rasterizer、未降低验收标准；普通学生质量评价完整保留，当前融合部署对该场景尚未通过原门限。','']
    lines += ['## 实际成本、失败与完整性','',
        'REPLICA_DISTILLATION_TRAINING.csv列出各阶段实际更新、图像反传次数、墙钟、峰值显存和复用标记。共享GPU下这些时间用于描述实验成本，速度对比待GPU空闲。渲染图片保存数为0，报告曲线单独导出。','']
    handoff_path = REPORTS/'PARALLEL_HANDOFF.json'
    if handoff_path.exists():
        handoff = read(handoff_path)
        lines += [f'运行中按用户要求从串行切换为最多{handoff["max_workers"]}组场景并发；同一场景四阶段仍顺序执行，训练配置与算法源码保持不变。切换时间和进程身份见PARALLEL_HANDOFF.json，切换前协议和状态保存在serial_handoff。并发阶段各任务墙钟包含资源竞争，不能将任务时间之和视作批次经过时间，也不能据此推算独占GPU加速。','',
            '切换时保留已运行的训练子进程，接管任务按完整训练CSV、完成标记及checkpoint核验成功；该旧子进程未观察到退出码，状态明确记录exit_code_observed=false。新启动任务记录实际退出码。互斥锁、已完成结果保留、场景与进程唯一性、算法哈希、阶段配置和原子状态写入检查见PARALLEL_VALIDATION.json。','']
    for job in state['scenes']:
        if job['status']=='failed':
            lines.append(f'- {job["id"]}失败原因为 {job.get("error")}。')
        for task in job.get('tasks',[]):
            if task.get('oom'):
                lines.append(f'- {job["id"]} {task["name"]} 第{task["attempt"]}次发生OOM，日志为 `{task["log"]}`。')
    integrity_path = REPORTS/'FINAL_INTEGRITY.json'
    if integrity_path.exists():
        integrity = read(integrity_path)
        for name,value in integrity.items():
            if isinstance(value,dict) and isinstance(value.get('files'),int):
                lines.append(f'- {name}核验 {value["files"]} 个文件，变更数 {len(value["changed"])}。')
        lines += ['',f'最终完整性核验通过状态为 {integrity["passed"]}，渲染图片保存数为 {integrity["saved_render_images"]}。','']
    lines += ['## 适用范围与产物','',
        '这是16个场景/相机组、单seed的固定Teacher蒸馏。每个Student使用自身Feature表坐标系。Teacher质量等价须按逐条件门限判断，均值无法排除局部长尾。全部流程保持原网络、正常训练与Rasterizer代码，优化发生在独立蒸馏中。','',
        '服务器根目录为 `/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108`。模型保存在 `models/Replica__<scene>__<mode>/continuous_calibration/best.pt`。汇总CSV为REPLICA_DISTILLATION_METRICS.csv，各组逐视角指标、属性与身份记录位于reports目录。复用组的原模型路径同时保留在身份JSON中。','',
        '下一步优先固定相同Teacher、选点、相机和初始化，比较32/32学生的全局与局部图像梯度方向和各属性误差，确定局部质量损失是否与目标冲突相关；随后只调整得到证据支持的一项蒸馏机制。room2单目的融合部署另需定位Scale/Rotation微扰触发的渲染敏感条件，保留原70dB门限和失败用例。当前未测量跨场景部署速度，不能将两路并发吞吐改善作为学生推理加速。','',
        '![单目PSNR](REPLICA_PSNR_MONO.png)','', '![RGB-D PSNR](REPLICA_PSNR_RGBD.png)','',
        '![单目差距](REPLICA_GAP_MONO.png)','', '![RGB-D差距](REPLICA_GAP_RGBD.png)','']
    report = '\n'.join(lines)
    (REPORTS/'REPLICA_COMPENSATOR_DISTILLATION_REPORT.md').write_text(report,encoding='utf-8')
    if ROOT==HERE/'results':
        for name in ['REPLICA_PSNR_MONO.png','REPLICA_PSNR_RGBD.png','REPLICA_GAP_MONO.png','REPLICA_GAP_RGBD.png']:
            report = report.replace('('+name+')','(results/reports/'+name+')')
        (HERE/'REPLICA_COMPENSATOR_DISTILLATION_REPORT.md').write_text(report,encoding='utf-8')
    print('REPLICA_REPORT',completed,failed,len(data),flush=True)


if __name__=='__main__':
    main()
