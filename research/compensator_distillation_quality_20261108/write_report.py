"""从实际日志生成中文技术报告与完整条件表。"""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st


HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results' if (HERE/'results').is_dir() else HERE.parents[2]
REPORTS = RESULTS/'reports'
SERVER = '/home/prmi/zwj/experiments/compensator_distillation_quality_20261108'
NAMES = dict(previous='上一版学生', control='原始残差权重1', residual001='原始残差权重0.01',
    anchor_refine='固定Ratio校准', continuous_refine='连续Ratio校准')


def read(name):
    return json.loads((REPORTS/name).read_text())


def main():
    result = read('FINAL_QUALITY_SUMMARY.json')
    phase1 = read('QUALITY_SUMMARY.json')
    perception = read('PERCEPTUAL_QUALITY.json')
    recheck = read('DEV_RECHECK_SUMMARY.json')
    selected = result['selected']
    index = {(r['student'], r['ratio'], r['local']):r for r in result['summaries']}
    perceptual = {(r['student'], r['ratio'], r['local']):r for r in perception['summaries']}
    final = [r for r in result['summaries'] if r['student'] == selected]
    previous = [r for r in result['summaries'] if r['student'] == 'previous']
    average_before = st.mean(r['mean_loss'] for r in previous)
    average_after = st.mean(r['mean_loss'] for r in final)
    mse_reduction = 1-st.mean(r['teacher_student_mse'] for r in final)/st.mean(r['teacher_student_mse'] for r in previous)
    parameters = 3534
    protocol = read('REFINEMENT_PROTOCOL.json')
    logs = {name:list(csv.DictReader((RESULTS/'models'/name/'training.csv').open()))
        for name in ['control', 'residual001', 'anchor_refine', 'continuous_refine']}
    exposure_keys = ['frame','ratio','local','K','full_K','image_ratio','image_local']
    same_exposure = all(a[k] == b[k] for a,b in zip(logs['control'], logs['residual001']) for k in exposure_keys)
    assert same_exposure
    source_hash = hashlib.sha256((HERE.parent.parent/'python/distill_compensation.py').read_bytes()).hexdigest()
    old_source = hashlib.sha256((RESULTS/'reference/python/distill_compensation.py').read_bytes()).hexdigest()
    phase1_source = read('PHASE1_SOURCE_MANIFEST.json')['python/distill_compensation.py']
    rows = []
    for record in result['summaries']:
        name = record['student']
        identity = result['identities'][name]
        metrics = perceptual.get((name, record['ratio'], record['local']), {})
        rows.append(dict(方案=NAMES[name],模式='局部' if record['local'] else '全局',ratio=record['ratio'],
            留出视角数=record['views'],教师PSNR=record['teacher_psnr'],学生PSNR=record['student_psnr'],
            平均PSNR损失=record['mean_loss'],p99损失=record['p99_loss'],最坏损失=record['worst_loss'],
            教师学生图像MSE=record['teacher_student_mse'],教师学生PSNR=record['teacher_student_psnr'],
            SSIM=metrics.get('ssim',''),LPIPS=metrics.get('lpips',''),严格门限通过=record['quality_gate'],
            MLP参数量=parameters,best步数=identity['step'],checkpointSHA256=identity['sha256'],
            源码SHA256=old_source if name == 'previous' else phase1_source if name in ['control','residual001'] else source_hash,
            checkpoint路径=identity['path']))
    with (REPORTS/'COMPENSATOR_DISTILLATION_QUALITY_RESULTS.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# Compensator 蒸馏质量优化报告', '',
        '## 结果', '',
        f'在 fr1_xyz RGB-D 的第二组独立留出视角上，22 个 Ratio×模式条件的平均 PSNR 损失从 {average_before:.5f} dB 降至 {average_after:.5f} dB，缩小 {(1-average_after/average_before)*100:.2f}%。教师与学生图像 MSE 的宏平均降低 {mse_reduction*100:.2f}%。这里的 PSNR 损失为相对同一个 GT 的“教师 PSNR − 学生 PSNR”，正数表示学生退化。', '',
        f'最终候选为 {NAMES[selected]}，检查点取第 {result["identities"][selected]["step"]} 步。学生沿用 32/32 主干和四个直接线性 Head，共 {parameters} 个 MLP 参数，使用已有 v31 专用 CUDA 路径。新增推理参数和算子均为 0。', '',
        f'沿用平均损失≤0.01 dB、p99≤0.05 dB、最坏≤0.1 dB 的严格门限，最终通过 {sum(r["quality_gate"] for r in final)}/22 个条件。通过项为全局 40%、60%、70%。当前结果适合作为进一步部署验证的候选，完整的教师质量保持目标尚未达成。', '',
        '## 误差诊断', '',
        '在验证视角 0、395、790 上对 6 个训练 Ratio、全局和局部模式测量 36 组梯度，诊断阶段参数更新次数为 0。原始 14D 归一化残差监督的梯度范数中位数为 3.786，XYZ 属性项为 0.00687，Scale 属性项为 0.02127。按原图像更新频率和权重折算后，图像梯度与总属性梯度范数之比的中位数为 0.230。11/36 组总属性梯度与图像梯度方向冲突。', '',
        '把单个学生 Head 输出替换为教师输出后，图像 MSE 下降中位数在全局/局部分别为 XYZ 51.6%/55.8%、Scale 37.5%/38.9%。Rotation 和 Other 的影响较小，各项包含属性间的交互。证据支持优先调整几何相关蒸馏监督与图像目标的平衡。', '',
        '第一阶段弱化残差监督后，固定训练 Ratio 的质量有所改善，3% 等区间内条件仍具有较高教师图像误差。第二阶段据此增加区间内教师采样，检验固定六档训练对学生连续映射的约束是否不足。', '',
        '## 蒸馏目标与梯度路径', '',
        r'最终使用 $\mathcal L=0.01\mathcal L_{res}+\mathcal L_{xyz}+\mathcal L_{scale}+\mathcal L_{rot}+\mathcal L_{opacity}+\mathcal L_{dc}+200\mathcal L_{image}$。', '',
        '原始残差项沿用固定 target_scale 归一化。XYZ 按 Anchor reference 尺度归一化，Scale 比较解码后的物理 Log-Scale，Rotation 使用符号不变的四元数距离，Opacity 和 SH DC 比较激活后的属性。图像项使用同一选点集合、同一相机、原分辨率下学生与教师 RGB 的 MSE。', '',
        '训练与验证图像 MSE 使用未 clamp 的 RGB，评价 PSNR、SSIM、LPIPS 使用 clamp 到 [0,1] 的 RGB。所有监督只更新学生私有 MLP 参数。Teacher、Selector、Master、Anchor Feature、reference 和归一化状态均作为 detach 输入。训练结束检查教师张量内容哈希和梯度状态。教师提供选点与补偿目标，GT 仅用于评价。', '',
        '每个训练步骤同时进行属性监督和完整图像监督。全局/局部模式交替，学生沿用对应的 ratio3、view7+mode1 条件和原始输出解码。', '',
        '## 两阶段匹配实验', '',
        '| 阶段 | 方案 | 每组 Adam 更新 | 图像更新 | 学习率 | 唯一处理因素 |',
        '|---|---|---:|---:|---:|---|',
        '| A | control / residual001 | 6000 | 6000 | 0.0002 | 原始残差权重 1 / 0.01 |',
        '| B | anchor_refine / continuous_refine | 3600 | 3600 | 0.0001 | 固定 Ratio / 50% 区间随机 Ratio |', '',
        '阶段 A 两组从上一版同一个 best.pt 初始化，使用相同随机种子、视角、选点与点数。日志逐行核对 frame、ratio、mode、K、full_K 和图像条件全部一致。两组统一按 3 个验证视角、6 个训练 Ratio、两种模式的教师图像 MSE 选择检查点。', '',
        '阶段 A 先执行 2400 步，在查看第一组新留出测试之前，依据验证误差仍在下降的结果将两组预算同时扩展到 6000 步。最终验证 MSE 为 control 9.22710e-5、residual001 6.43509e-5，后者低 30.26%。相对上一版同时改变了学习率、图像更新频率和检查点选择口径，这部分收益由匹配对照区分。', '',
        '阶段 B 两组从 residual001 的第 5400 步共同初始化。固定档为 0.01、0.05、0.1、0.3、0.5、0.7。连续组以 50% 概率保留轮转档位，以 50% 概率等概率选择相邻档位区间，并在该区间按 log(ratio) 均匀采样。全局和局部的同一 Ratio 对共享该采样值。', '',
        '视角使用 seed 和步号独立生成，防止不同 K 对 randperm 消耗的差异改变后续视角序列。两组日志核对 3600 个 frame 和 mode 全部一致。Ratio 采样同样由 seed 和 Ratio 对序号确定，恢复训练时可重建采样值。', '',
        '阶段 B 的验证 Ratio 为六档加五个相邻档位的几何中点，两组共用这 11 个 Ratio×3 个视角×2 种模式。固定组最佳 MSE 为 8.81100e-5，连续组为 7.60146e-5，降低 13.73%。跨阶段验证集合不同，验证分数按阶段内比较。', '',
        '| 方案 | 实际属性更新 | Full 分辨率图像更新 | 属性分支点数累计 | 图像分支点数累计 |',
        '|---|---:|---:|---:|---:|'
    ]
    for name, history in logs.items():
        text.append(f'| {NAMES[name]} | {len(history)} | {len(history)} | {sum(int(r["K"]) for r in history)} | {sum(int(r["full_K"]) for r in history)} |')
    text += ['', '正式实验共 19200 次学生 Adam 更新，另外执行 8 次恢复功能测试更新。阶段 B 的 Ratio 数据不同，实际点数累计随之变化，更新次数、视角和图像分辨率一致。最终模型的有效权重路径为旧学生 6500 步＋阶段 A 5400 步＋阶段 B 3000 步，共 14900 步，采用两次独立校准的 Adam 初始化。', '',
        '## 留出评价与边界', '',
        '学生训练只使用帧序号不能被 5 整除的视角。验证视角为 0、395、790。历史 16 帧与第一组 32 帧在方案迭代后作为开发数据。最终结果使用预先记录的另一组 32 帧，和上述数据、学生训练视角均互斥。留出专指学生蒸馏视角，教师和地图来自同一场景。完整帧列表保存在 REFINEMENT_PROTOCOL.json。', '',
        '最终先按验证教师图像 MSE 确定候选，再执行 32×11×2×5=3520 组配对比较。局部模式逐组核对学生与教师选择身份一致，全局模式复用教师选点集合。实际 K 在 FINAL_QUALITY_FRESH.csv 中逐组记录。局部保持原严格 K，全局沿用原有阈值与同分语义。', '',
        '训练期间区间内随机 Ratio 已获得教师监督。因此 0.03、0.08、0.2、0.4、0.6 在最终表中标为“非锚点 Ratio”，它们表示具体测试值位于锚点之间。', '',
        '### 平均 PSNR 损失', '',
        '| Ratio | 类型 | 全局旧版 | 全局最终 | 局部旧版 | 局部最终 |',
        '|---:|---|---:|---:|---:|---:|']
    for ratio in sorted(set(r['ratio'] for r in final)):
        a,b,c,d = [index[(n,ratio,l)]['mean_loss'] for n,l in [('previous',False),(selected,False),('previous',True),(selected,True)]]
        text.append(f'| {ratio:g} | {"锚点" if ratio in [.01,.05,.1,.3,.5,.7] else "非锚点"} | {a:.5f} | {b:.5f} | {c:.5f} | {d:.5f} |')
    text += ['', '### 最终模型的尾部', '', '| Ratio | 全局 p99 / 最坏 | 局部 p99 / 最坏 |', '|---:|---:|---:|']
    for ratio in sorted(set(r['ratio'] for r in final)):
        g,l = [index[(selected,ratio,v)] for v in [False,True]]
        text.append(f'| {ratio:g} | {g["p99_loss"]:.5f} / {g["worst_loss"]:.5f} | {l["p99_loss"]:.5f} / {l["worst_loss"]:.5f} |')
    text += ['', '第一组 32 帧在最终模型固定后再次检查，1% 局部损失从上一版 0.01562 dB 变为 0.01791 dB，阶段 A 为 0.03107 dB。第二组独立 32 帧则从 0.02418 变为 0.02176 dB。1% 的改善方向在两组间不一致，当前证据支持其误差回到接近旧版的范围，稳定质量收益仍需确认。', '',
        '连续采样主要改善区间内拟合。相对阶段 A 弱残差模型，部分 30%—70% 局部条件的 GT PSNR 损失略有增加，全部模型的并列表位于结果 CSV。候选选择只使用教师图像误差，教师拟合程度与 GT 质量指标之间存在差异。', '',
        '### SSIM 与 LPIPS', '',
        '同一最终 32 帧上的 SSIM 在 21/22 个条件提高，LPIPS 在 17/22 个条件降低。宏平均 SSIM 增加 0.000882，LPIPS 降低 0.000692。LPIPS 轻微回退出现在局部 1%、3%、20%、30%、40%，最大增量为 1% 的 0.000782。', '',
        '| Ratio | 全局 SSIM 旧→新 | 全局 LPIPS 旧→新 | 局部 SSIM 旧→新 | 局部 LPIPS 旧→新 |',
        '|---:|---|---|---|---|']
    for ratio in sorted(set(r['ratio'] for r in final)):
        values=[]
        for local in [False,True]:
            a,b=[perceptual[(n,ratio,local)] for n in ['previous',selected]]
            values.extend([f'{a["ssim"]:.5f}→{b["ssim"]:.5f}',f'{a["lpips"]:.5f}→{b["lpips"]:.5f}'])
        text.append('| '+str(ratio)+' | '+' | '.join(values)+' |')
    parity = read('TRAINED_CUDA_PARITY.json')
    text += ['', '## 实现与正确性验证', '',
        '实际产品源码改动集中在 python/distill_compensation.py。新增属性监督重权、图像验证口径、独立验证 Ratio、区间 Ratio 采样和逐步独立视角采样。对应参数为 --residual-weight、--validation-metric、--validation-ratios、--ratio-jitter、--paired-view-sampling。上述参数写入 checkpoint 元数据，恢复训练时严格检查配置。', '',
        '同一真实输入下，旧默认属性目标与原表达式逐位一致。原始残差重权的参数梯度与显式公式逐位一致。image_mse 验证分数与手工渲染 MSE 一致，独立于属性权重。', '',
        'Ratio 采样覆盖全部五个相邻区间，同一全局/局部 Ratio 对采样一致，与外部 RNG 消耗无关。连续采样模型完成保存和恢复到第 4 步，历史步号连续，Adam 状态完整加载。改变残差权重、Ratio 分布或视角采样方式的错误恢复被拒绝。', '',
        f'最终权重共验证 {parity["cases"]} 组 CUDA/PyTorch 学生输出，包括全局和局部、六档和五个非锚点 Ratio。14D 输出最大绝对误差为 {parity["max_residual_abs"]:.9g}，使用 rtol=1e-4、atol=1e-5。两种学生实现渲染的最低相互 PSNR 为 {parity["min_image_psnr"]:.2f} dB。55 个局部请求的 Graph/Native 及选择身份共 495 次逐位比较全部通过。', '',
        '本轮复用已验收的 v31 部署二进制，GPU Kernel 源码和网络布局由既有二进制执行。Python 源码编译、目标测试、采样测试、保存恢复、实际图像反传与部署检查全部通过。数值评价中的属性和图像均通过非有限值检查。', '',
        '### 几何偏差', '',
        '逐 case 计算归一化 XYZ 位移 p99 和学生/教师物理 Scale 比值 p99。704 个最终 case 的 p99 中位数中，XYZ 位移从 0.39230 变为 0.39092，XYZ 误差从 0.13227 降至 0.12626，Scale 比值从 1.12492 降至 1.09541。所测 case 的最大归一化位移从 9.02790 变为 8.80432，最大 Scale 比值从 1.78959 变为 1.59729。这些统计衡量学生相对教师的偏差，点级可见性和贡献度影响另需逐点分析。', '',
        '## 性能与完整性', '',
        '学生 MLP 参数为 3534，教师为 25742。本轮新增质量优化发生在离线训练目标和训练样本分布中，部署使用同一套学生网络与 CUDA 算子。服务器存在其他 GPU 进程，本轮速度测试状态为待空闲 GPU 测量。理论参数量与 MAC 变化沿用前轮记录，当前报告只据实给出质量结果。', '',
        '研究前后 604 个生产源码、配置、二进制和动态库文件哈希一致，84 个既有模型输入文件及上一版学生 checkpoint 哈希一致。部署二进制 SHA256 为 fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601。Rasterizer 二进制 SHA256 为 27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86。', '',
        '渲染图片保存数为 0。报告曲线单独导出为 PNG。工作树补丁、参考源码、SHA 清单、配置、模型与完整日志构成本轮可追溯交付。', '',
        '## 范围与剩余问题', '',
        '结论来自单场景、单 seed 的冻结地图学生蒸馏。32 个最终留出视角仍属于同一条相机轨迹。结果说明重新平衡监督和增加 Ratio 区间采样能够改善当前学生拟合，学生容量上限和不同场景的迁移收益尚未隔离。', '',
        '1% 的稳定收益、局部模式的 PSNR 长尾和少数 LPIPS 回退仍未解决。严格质量门限仅通过 3/22，教师替代的最终质量要求仍有差距。', '',
        '## 交付位置', '',
        f'- 服务器研究目录为 `{SERVER}`。',
        f'- 最终学生为 `{result["identities"][selected]["path"]}`。',
        f'- 最终学生 SHA256 为 `{result["identities"][selected]["sha256"]}`。',
        '- 完整汇总为 COMPENSATOR_DISTILLATION_QUALITY_RESULTS.csv，逐视角结果为 FINAL_QUALITY_FRESH.csv。',
        '- 独立对照与过程保存在 QUALITY_SUMMARY.json、REFINEMENT_STATUS.json、REFINEMENT_PROTOCOL.json、DEV_RECHECK_SUMMARY.json。', '',
        '![最终留出集的平均PSNR差距](COMPENSATOR_DISTILLATION_QUALITY.png)', '',
        '![匹配校准的验证误差](TRAINING_IMAGE_MSE.png)', '']
    report = '\n'.join(text)
    if (HERE/'results').is_dir():
        entry = report.replace('(COMPENSATOR_DISTILLATION_QUALITY.png)', '(results/reports/COMPENSATOR_DISTILLATION_QUALITY.png)')
        entry = entry.replace('(TRAINING_IMAGE_MSE.png)', '(results/reports/TRAINING_IMAGE_MSE.png)')
        (HERE/'COMPENSATOR_DISTILLATION_QUALITY_REPORT.md').write_text(entry, encoding='utf-8')
    (REPORTS/'COMPENSATOR_DISTILLATION_QUALITY_REPORT.md').write_text(report, encoding='utf-8')
    print('REPORT_COMPLETE', len(rows))


if __name__ == '__main__':
    main()
