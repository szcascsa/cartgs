from pathlib import Path
import csv
import json
import statistics


ROOT=Path(__file__).resolve().parents[3]


def read(name):return json.loads((ROOT/'reports'/name).read_text())


def main():
    first=read('fr1_quality_independent/quality.json')
    final=read('fr1_image_quality/quality.json')
    assert first['frames']==final['frames'] and first['teacher_fingerprint']==final['teacher_fingerprint']
    initial={(r['ratio'],r['local']):r for r in first['summaries']}
    rows=[]
    for current in final['summaries']:
        before=initial[current['ratio'],current['local']]
        rows.append(dict(ratio=current['ratio'],local=current['local'],views=current['views'],
            attribute_stage_psnr_loss=before['mean_gt_psnr_loss'],
            calibrated_psnr_loss=current['mean_gt_psnr_loss'],
            teacher_student_image_psnr=current['teacher_student_psnr'],
            p99_loss=current['p99_gt_psnr_loss'],worst_loss=current['worst_gt_psnr_loss'],
            quality_gate_passed=current['quality_gate_passed']))
    with (ROOT/'reports/COMPENSATOR_DISTILLATION_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gradients=read('DISTILLATION_GRADIENTS.json')
    gradient_ratio=statistics.median(r['photo_to_attribute_gradient'] for r in gradients['rows'])
    cpu=read('CPU_STUDENT_TEST.json');gpu=read('GPU_STUDENT_TEST.json');parity=read('TRAINED_CUDA_PARITY.json')
    identity=read('BINARY_IDENTITY.json');integrity=read('FINAL_INTEGRITY.json')
    assert integrity['passed'] and read('DELIVERY_TESTS.json')['complete']
    lines=['# Compensator32 蒸馏实现与验证报告','',
        '已实现独立学生训练、校准初始化、保存恢复、质量评价与冻结 CUDA 部署。本轮只蒸馏 Compensator，Selector、Master、Anchor Feature 及拓扑采用原 checkpoint 的冻结状态。实际速度尚待 GPU 空闲后的配对测量。','',
        '当前学生尚未通过沿用的严格质量门槛，所有正式比较均采用同一个教师模型和同一套选点。代码与学生权重作为独立实验交付。','',
        '## 学生结构','',
        '|项目|Teacher|Student|', '|---|---|---|',
        '|物理及 Feature 输入|Offset3＋Residual48，51D|相同|',
        '|连续 ratio|独立 3D 投影|相同|',
        '|局部视角|7D＋模式1D 的独立投影|相同|',
        '|共享主干|64、64|32、32|',
        '|四个 Head|各自 64→64→属性输出|各自 32→属性输出|',
        '|最长 Linear 路径|4层|3层|',
        '|MLP 参数，含视角投影|25,742|3,534|',
        '|每点线性层 MAC，含视角投影|25,344|3,456|','',
        '学生网络参数减少约 86.27%，MAC 减少约 86.36%。这些数字不包含冻结 Anchor Feature 表、Selector、Master、Feature 查询、非线性、解码和选点。主地图和 Feature 表的体积仍需独立计算，不能由 MLP 压缩率推断整个 checkpoint 的压缩率或整帧加速。','',
        'Residual48 的 Base 与活跃 Residual 按原顺序查询。XYZ 输出继续乘 Anchor 参考尺度，Scale、四元数、Opacity 和 SH DC 沿用对应教师 checkpoint 的物理变换。Teacher 的 Softplus 与物理加法两种 Scale 解码均可读取。','',
        '## 蒸馏目标和梯度路径','',
        '学生输入、Master 属性、Feature 表、教师目标和参考尺度均 detach。仅学生的 13 个参数 Tensor 及可选视角投影参与 Adam。继承教师前 32 个隐藏神经元用于初始化共享层，再对训练视角的校准样本做 FP64 Ridge 输出层拟合。校准完成后的训练和部署采用 FP32。','',
        '监督包括归一化14维残差 MSE、参考尺度归一化 XYZ、物理 Log-Scale、旋转四元数的符号等价距离、激活 Opacity、SH DC，以及同一选点集合下的教师图像 MSE。旋转项为 1−⟨qS,qT⟩²，避免把 q 与 −q 当作不同旋转。加法 Scale 的无效负值在损失中有恢复项，渲染入口继续拒绝非有限属性。','',
        '训练同时轮转全局模式和局部模式。全局模式复用当前工程的 FlexGS 阈值及同分规则，局部模式复用名义 Anchor 候选池与严格 K。图像监督使用完整选择集合，属性小批采样只减少每步监督点数。图像的 ratio／模式独立轮转，避免低频间隔与属性周期重合后始终只训练某一种模式。','',
        '## 实际训练与质量修正','',
        '第一阶段执行 6,000 个 Adam 更新，每次属性样本最多 4,096 点，学习率 0.001，seed=0，训练 ratio 为 0.01、0.05、0.1、0.3、0.5、0.7。每 20 步加入一次权重 1 的完整图像监督，共 300 次。按属性验证分数选择的最佳 checkpoint 位于第 '+str(first['student_step'])+' 步。','',
        f'单个匹配验证视角的 12 组梯度诊断中，图像梯度与属性梯度的范数比中位数约为 {gradient_ratio:.6f}。这说明第一阶段图像监督明显偏弱，尚不能把质量差距全部归因于学生容量。诊断只在一个视角执行，方向、范数与频率折算值均保存在 DISTILLATION_GRADIENTS.json。','',
        '第二阶段从第一阶段最佳权重初始化，重建 Adam 和采样 RNG，增加 2,000 步图像校准。图像权重改为 1000，每 5 步一次，共 400 次。验证分数加入权重 1000/5 的匹配图像 MSE，使最佳权重的选择与实际训练目标相符。该阶段同时改变图像权重、频率和 checkpoint 选择指标，不能据此拆分三者各自的收益。','',
        '最终最佳权重来自校准阶段第 '+str(final['student_step'])+' 步，沿权重继承路径共有 '+str(first['student_step']+final['student_step'])+' 次更新。两阶段实际执行总预算为 8,000 步。源码保留第一阶段快照 STAGE1_SOURCE.zip，训练配置分别保存在两个模型目录。','',
        '最终训练入口默认图像权重为 1000、间隔为 5，全部训练参数均可通过 CLI 配置。--resume 恢复权重、Adam 和采样状态并检查目标一致性，--initialize 只继承权重并开始新训练。旧目标的 checkpoint 用于初始化新目标时需明确选择 --initialize。','',
        '## 逐 ratio 质量','',
        '评估覆盖 16 个独立测试视角、11 个 ratio、全局与局部两种模式，共 352 组。每五帧留出一帧，训练排除全部留出帧。测试进一步排除用于选择 checkpoint 的三个验证视角。两个阶段使用完全相同的测试视角及教师选点。评价原图全分辨率，图片保存在内存中，磁盘图片数为零。','',
        'PSNR 损失表示 Teacher 对 RGB 真值的 PSNR 减去 Student 对同一真值的 PSNR，正值为下降。未见 ratio 为 0.03、0.08、0.2、0.4、0.6。质量门槛预先沿用平均损失不超过 0.01 dB、p99 不超过 0.05 dB、最差不超过 0.1 dB，逐 ratio、逐模式判断。','',
        '|Ratio|模式|属性阶段平均损失 dB|图像校准后平均损失 dB|Student 对 Teacher 图像 PSNR dB|门槛|',
        '|---|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"|{row['ratio']}|{'局部' if row['local'] else '全局'}|{row['attribute_stage_psnr_loss']:.4f}|{row['calibrated_psnr_loss']:.4f}|{row['teacher_student_image_psnr']:.3f}|{'通过' if row['quality_gate_passed'] else '未通过'}|")
    lines += ['', '所有逐视角 Teacher／Student PSNR 和属性损失保存在两个 quality.csv。当前结果仅来自 fr1_xyz 的单个教师 checkpoint，尚无跨场景蒸馏结论。','',
        '## CUDA 与现有优化兼容','',
        'Student CUDA Kernel 在原有冻结部署调度中根据 13 个参数 Tensor 的结构分发。Teacher 的 21 Tensor 分支保持独立。学生专用 4／8／16 点 Tile 在同一次 Kernel 中完成 Feature Gather、顺序 Residual 累加、输入打包、32/32 MLP、四个输出 Head、物理解码与发布。比例三角函数仍按请求计算一次。','',
        '继续复用 Selector 静态前缀、每帧可见 Anchor 编码、GPU 动态 N/K、稳定选择、有限值状态、Graph 计划、输出独立存储和大地图 Host 计数路径。Student 作为显式选项传入，正常训练入口与原 Teacher checkpoint 均保留原有身份。','',
        '学生 checkpoint 绑定 Teacher、Master、Feature 表、Anchor 映射和 Gaussian 行顺序的 Tensor 内容 SHA256。地图、权重或拓扑变化时需要重新训练／绑定并重建部署实例。加载时检查结构、有限值、视角 schema 和教师签名。','',
        '## 测试结果','',
        '|验证|结果|', '|---|---|',
        f"|固定输入 Student CUDA 对 PyTorch|{gpu['cases']}组通过，最大残差误差 {gpu['max_residual_error']:.3g}|",
        f"|训练后权重 CUDA 对 PyTorch|{parity['cases']}组通过，最大残差误差 {parity['max_residual_error']:.3g}，最低图像对比 PSNR {parity['min_fused_image_psnr']:.3f} dB|",
        '|Teacher 原路径|与 v30 的选点和属性逐位一致|',
        '|Graph 与 Native|Student 选点和属性逐位一致|',
        '|五类 checkpoint schema|真实视角、仅模式、单分支视角和旧11D均通过|',
        '|CPU 梯度隔离|Teacher、Master、输入、Feature无梯度，仅 Student 更新|',
        '|Checkpoint／Adam|保存恢复与 CPU 下一步更新逐位一致，错误教师签名拒绝加载|',
        '|真实入口|训练、渲染梯度、保存、恢复、独立测试视角评价通过|',
        '|Compute Sanitizer|memcheck、initcheck 零错误|','',
        f"CPU 合成样本训练损失从 {cpu['initial_training_loss']:.6f} 降为 {cpu['final_training_loss']:.6f}，合成留出损失从 {cpu['initial_validation_loss']:.6f} 变为 {cpu['final_validation_loss']:.6f}。该单元测试检验优化器及隔离语义，留出误差上升已保留记录，真实质量结论采用上面的独立场景评价。",'',
        'Teacher 与 Student 结构不同，因此不要求两者逐位相同。Student 的 CUDA 与自身 PyTorch 使用独立的数值容差与图像对比验收。推理误差与蒸馏误差据此分别评估。','',
        '## 服务器空闲后测速','',
        '本轮没有推理速度数字。benchmark 入口检查其他 CUDA 进程，GPU 未空闲时拒绝开始，并在每个条件前后复核。采用共同预热、ABBA 顺序、三个重复、跳转相机和实际 N/K，输出 prepare 与 prepare 加原 Rasterizer 的配对数据。MAC 压缩率不作为实测加速。','',
        '在本研究目录运行以下命令。所有路径指向独立研究输出。','',
        '```bash',
        'cd '+str(ROOT),
        'PY=/home/prmi/anaconda3/envs/zwj/bin/python',
        'ENTRY=source/research/compensator_distillation_20261108/run.py',
        'TEACHER=/home/prmi/zwj/experiments/view_conditioned_li_20261106/runs/full_both_li/23560_shutdown/ply',
        'DATA=/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
        '$PY "$ENTRY" benchmark "$TEACHER" --dataset "$DATA" --student models/fr1_student32_image_calibration/best.pt --output reports/idle_speed',
        '```','',
        '重新蒸馏使用 train 子命令；延续同目标使用 --resume；从已有 Student 开始新的目标使用 --initialize。评价使用 evaluate 子命令与 --student。直接部署入口支持 DeploymentRenderer.from_checkpoint(teacher, student_checkpoint=...)，deployment_infer.py 支持 --student。','',
        '## 完整性与限制','',
        '|对象|核验文件数|变化数|', '|---|---:|---:|']
    for tag in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        d=integrity[tag];lines.append(f"|{tag}|{d['files']}|{len(d['changed'])}|")
    lines += ['', '新扩展 '+identity['binary']['path']+'，SHA256 为 '+identity['binary']['sha256']+'。', '',
        '生产源码、已有模型、共享环境与 Rasterizer 采用既有只读输入，Git HEAD 保持本轮起点，未执行 commit／push。学生权重、数值报告和独立二进制保存在本轮目录。','',
        '下一步先在空闲 GPU 测真实收益。如果质量仍未达到门槛，优先用同一初始化、预算及测试视角，分别控制图像权重、图像频率和最佳 checkpoint 指标，并检查局部模式的属性残差长尾。当前已验证的压缩与数值兼容性不等同于教师质量可替代性。']
    (ROOT/'reports/COMPENSATOR_DISTILLATION_IMPLEMENTATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT_READY',final['all_quality_gates_passed'],flush=True)


if __name__=='__main__':main()
