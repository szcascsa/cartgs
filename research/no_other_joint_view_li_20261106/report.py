"""依据已完成的统计生成中文报告，不选择最佳轮次或训练终点。"""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'results';OUT=ROOT/'evaluation/reports'
SERVER='/home/prmi/zwj/experiments/no_other_joint_view_li_20261106'
CONTROL='/home/prmi/zwj/experiments/view_conditioned_li_20261106'


def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))


def main():
    s={(r['group'],r['case']):r for r in load('evaluation/reports/SUMMARY.json') if r['partition']=='all'}
    p={r['case']:r for r in load('evaluation/reports/PAIRED_SUMMARY.json') if r['partition']=='all'}
    speed={(r['group'],r['budget']):r for r in load('speed/SUMMARY.json') if r['repeat']=='all'}
    paired=load('speed/PAIRED_SUMMARY.json');params=load('speed/PARAMETERS.json');gate=load('evaluation/reports/QUALITY_GATES.json')
    micro=load('speed/COMPENSATION_SAME_INPUT_SUMMARY.json');back=load('speed/COMPENSATION_FORWARD_BACKWARD_SUMMARY.json')
    geometry={(r['group'],r['case']):r for r in load('evaluation/reports/GEOMETRY_SUMMARY.json')}
    lines=[]
    def add(text=''):lines.append(text)
    def table(headers,values):
        add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
        for row in values:add('| '+' | '.join(map(str,row))+' |')
        add()
    add('# 当前默认架构移除Other Head：重训、速度与质量对照\n')
    add('## 结论\n')
    for budget,label in [('points500','局部500点'),('ratio50','局部50%')]:
        a,b=speed['four_head',budget],speed['no_other',budget]
        saved=100*(a['total_ms_mean']-b['total_ms_mean'])/a['total_ms_mean']
        case='points_500' if budget=='points500' else 'view_0.5'
        add(f'- {label}：含推理与渲染总耗时 {a["total_ms_mean"]:.4f}→{b["total_ms_mean"]:.4f} ms，耗时变化 {-saved:+.2f}%；PSNR {s["four_head",case]["psnr_mean"]:.4f}→{s["no_other",case]["psnr_mean"]:.4f} dB（{p[case]["delta_psnr_mean"]:+.4f} dB）。')
    add(f'- 六个训练ratio的全局平均ΔPSNR为 {gate["global_train_mean_delta"]:+.4f} dB，最差一档 {gate["global_train_min_delta"]:+.4f} dB；Full变化 {gate["full_delta"]:+.4f} dB。预声明全局质量保护：**{"通过" if gate["quality_guard_passed"] else "未通过"}**。局部低ratio须独立判断。')
    a,b=geometry['four_head','view_0.01'],geometry['no_other','view_0.01']
    add(f'- 局部1%下XYZ／参考尺度p99的视角均值 {a["xyz_p99_mean"]:.3f}→{b["xyz_p99_mean"]:.3f}，位移长尾增大。结合低ratio质量下降与未证明整体提速，本轮结果不支持将无Other方案替换为生产默认。')
    add('- 这是fr1_xyz RGB-D、单seed的固定拓扑Fine消融，质量变化包含联合训练后Master、Selector和Feature共同适应。不能外推为所有场景结果或纯删除算子的数值等价。生产默认未切换。\n')
    add('## 实际改动与匹配条件\n')
    add('删除Compensator的Other分支`64→64→4`，减少4420参数；保留Residual48、64/64主干、XYZ3／Scale3／Rotation4三个独立Head、两网络真实视角和LI=0.1。14D接口末四维补常数零，Opacity与SH DC不再由Compensator补偿，但Master的Opacity/DC仍正常优化。未进一步删除14D解码中的加零操作，避免混入其他性能改动。\n')
    add('配置`Selector.geometry_head_variant: 1`，原四Head缺省为0；checkpoint记录`geometry_head_schema=1`，Python/C++保存恢复和冻结克隆均保留该含义。旧Anchor128及合并MLP分支保留。核心变更仅在`spatial_compensation.h`、`gaussian_mapper.h/.cpp`、`lod_from_full.cpp`和独立消融YAML。\n')
    add('另修正既有Python几何消融每次前向`int(GPU schema)`引入的同步：在configure/load时缓存不可变结构编号为Python整数，保留checkpoint buffer。缓存前后、原生fixture在CPU/CUDA的global/local共4组逐bit通过；未改Tensor算术和训练逻辑。四Head缺省分支仍为0。该补丁也保留在独立源码中。\n')
    table(['项目','匹配设置'],[
        ['源码基线','26c355b32bf373a75877fc4605a4e41b8de9b47f＋本轮无Other补丁；排除并行推理融合修改'],
        ['起点','与both_li相同Full11980；Master SHA256 26e605baec3195e23088adc0fe29a52e0d1efd9d072762839c6b23299ea4ead5'],
        ['训练','seed0，11580步，终点23560；Full→global LoD→local LoD，逐路反传后一次Adam'],
        ['ratio','0.01、0.05、0.1、0.3、0.5、0.7，各1930步'],
        ['地图','N=13298，Anchor=2790，拓扑固定；Master允许更新，72个训练关键帧'],
        ['曝光','Full/global/local各11580次；GI刷新12次，另864次Full；LI复用11580次已有Full'],
        ['优化器','fresh Adam/RNG；六Feature表时钟11580/9650/7720/5790/3860/1930；不是完整SLAM/Adam恢复'],
        ['其他','Softplus Scale启用，物理Scale加法关闭；无XYZ上限；Maturity/SGI关闭；训练不筛名义视锥'],
        ['推理','全局导出从全图选；局部预算沿用Viewer名义Anchor候选池，严格K；真实视角＋LI模型'],
        ['评价','同792帧，Bank72/nonBank720，640×480 FP32，TF32关闭；PSNR/SSIM11×11/LPIPS-AlexNet/photo-L1']])
    add('原四Head训练及已有34条件26928条结果只读复用；本轮补测旧500点并评价新模型35条件28512次，最终两模型共55440条逐帧记录。没有重复训练原四Head。matched条件按各模型自身global Alpha贡献点数请求局部预算，不是跨模型相同地图、相同点集对照。\n')
    add('## 正确性与验收\n')
    add('- CPU/CUDA各80种N／ratio／global-local前向：保留10D逐bit一致，Other恒零；共同输入XYZ/中心/参考尺度/Feature及保留参数VJP按名称逐bit一致。此VJP对比将四Head的Other上游梯度设零，仅证明保留路径实现正确，不要求重训过程梯度相同。')
    add('- 删除Head不消耗RNG；移除4420参数；checkpoint恢复、cloneFrozen、跨语言CPU/CUDA global/local共4组逐bit通过。LI只累加六个Selector私有参数的梯度，Master／Compensation／Feature既有梯度逐bit保持。')
    add('- 36步smoke后重新从原Full启动正式训练。63个共同状态Tensor、初始Master、首步Full/global/local/LI/budget/joint Loss及K一致；全部11580步相机和ratio序列一致。')
    add('- 原生健康检查保留，正式训练非有限计数0；真实Renderer与带贡献统计Renderer在首／中／末帧共108组图像与radii逐bit通过。')
    add('- 验收期间修正三个测试问题：删除并重新注册Head改变参数遍历顺序，改为按名称比较；fixture的`global`为TorchScript保留字，改名；Python/C++ CPU线程数统一为2后跨语言逐bit通过。均发生在正式训练前，失败日志保留，未放宽数值判据。\n')
    add('## 全轨迹质量\n')
    for scope,title in [('global','全局导出'),('view','局部视角预算')]:
        add(f'### {title}\n')
        values=[]
        for ratio in [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]:
            case=f'{scope}_{ratio:g}';a,b=s['four_head',case],s['no_other',case];d=p[case]
            values.append([f'{ratio:g}',f'{a["psnr_mean"]:.4f}',f'{b["psnr_mean"]:.4f}',f'{d["delta_psnr_mean"]:+.4f}',
                f'{d["delta_ssim_mean"]:+.5f}',f'{d["delta_lpips_mean"]:+.5f}',f'{100*d["psnr_improved_fraction"]:.1f}%',f'{b["K_mean"]:.1f}'])
        table(['ratio','四Head PSNR','去Other PSNR','ΔPSNR↑','ΔSSIM↑','ΔLPIPS↓','改善视角','新实际K均值'],values)
    add('### Full与固定局部预算\n')
    table(['条件','四Head PSNR','去Other PSNR','ΔPSNR','ΔSSIM','ΔLPIPS','实际K均值'],[
        [case,f'{s["four_head",case]["psnr_mean"]:.4f}',f'{s["no_other",case]["psnr_mean"]:.4f}',f'{p[case]["delta_psnr_mean"]:+.4f}',
         f'{p[case]["delta_ssim_mean"]:+.5f}',f'{p[case]["delta_lpips_mean"]:+.5f}',f'{s["no_other",case]["K_mean"]:.1f}']
        for case in ['full']+[f'points_{k}' for k in [500,1000,2000,3000,4000,5000,10000]]])
    add('训练ratio与未见ratio均完整评价，未据nonBank选择终点或权重。0.7属于训练ratio。全局保护门槛在运行前写入PROTOCOL：训练档ΔPSNR均值≥−0.10 dB、任一档≥−0.20 dB、Full≥−0.10 dB。Bank/nonBank、median/p10/p90、photo-L1、逐帧差异及实际K在CSV中，不能以均值掩盖低ratio退化。\n')
    add('## 速度、计算量与显存\n')
    add('同进程、同冻结PyTorch源码和相机，两个模型交错运行；64次预热后792帧×3轮，顺序／逆序／seed0乱序，全部14256条保留。含原有finite／状态检查、候选构建、选点、Feature和视角处理、补偿及Rasterizer；不含模型加载、GT、指标、HTTP或显示。下表为同步墙钟均值，Selector/Compensation沿用原模块计时，其他准备为差值。\n')
    table(['条件','Head','Selector ms','Comp ms','其他准备 ms','渲染 ms','总计 ms'],[
        [budget,group,f'{r["selector_ms_mean"]:.4f}',f'{r["compensation_ms_mean"]:.4f}',f'{r["other_prepare_ms_mean"]:.4f}',f'{r["render_ms_mean"]:.4f}',f'{r["total_ms_mean"]:.4f}']
        for budget in ['points500','ratio50','full'] for group in ['four_head','no_other'] for r in [speed[group,budget]]])
    table(['条件','轮次','总耗时减少','推理减少','Comp减少','实际K相同比例'],[
        [r['budget'],r['repeat'],f'{r["total_ms_reduction_pct"]:+.2f}%',f'{r["inference_ms_reduction_pct"]:+.2f}%',
         f'{r["compensation_ms_reduction_pct"]:+.2f}%',f'{100*r["same_K_fraction"]:.1f}%'] for r in paired if r['budget']!='full'])
    add('### 共同输入／共同保留权重的Compensation微测\n')
    add('将同一个已训练四Head Compensator深复制后删除Other，其他权重、输入、ratio和batch完全相同。前向每条件200对；前向＋反向每条件80对，交错顺序，CUDA Event及同步墙钟均记录。后者对所有14通道使用非零上游梯度，覆盖被删除分支对共享主干的反传成本，排除Feature表scatter、Photo/Rasterizer及Adam。\n')
    values=[]
    for mode,records in [('前向',micro),('前向＋反向',back)]:
        mi={(r['K'],r['ratio'],r['group']):r for r in records}
        for k in [500,4096]:
            for ratio in [.05,.5]:
                a,b=mi[k,ratio,'four_head'],mi[k,ratio,'no_other']
                values.append([mode,k,ratio,f'{a["cuda_stream_ms_mean"]*1000:.2f}',f'{b["cuda_stream_ms_mean"]*1000:.2f}',
                    f'{100*(1-b["cuda_stream_ms_mean"]/a["cuda_stream_ms_mean"]):+.2f}%',f'{a["wall_ms_mean"]*1000:.2f}→{b["wall_ms_mean"]*1000:.2f}'])
    table(['范围','K','ratio','四Head Event μs','去Other Event μs','耗时减少','同步墙钟μs'],values)
    add('CUDA Event微测衡量当前流完成时间，可能包含主机发射空隙或共享GPU等待；独立Profiler的kernel时间与launch在`speed/PROFILE_SUMMARY.json`，不与无Profiler的墙钟混算。重训前后地图与footprint可变，因此Renderer和整个推理变化并非全部由Head计算量造成。同输入微测更适合归因。\n')
    profile=load('speed/PROFILE_SUMMARY.json')
    table(['局部预算','Head','每次Kernel数','Kernel累计μs'],[
        [budget,g,f'{np.mean([r["kernels"] for r in profile if r["scope"]=="METHOD/"+g+"/"+budget]):.2f}',
         f'{np.mean([r["kernel_us"] for r in profile if r["scope"]=="METHOD/"+g+"/"+budget]):.2f}']
        for budget in ['points500','ratio50'] for g in ['four_head','no_other']])
    add('Profiler仅4个代表视角，用于调用归因，不能作为稳定速度均值。两种局部预算均仅减少2个Kernel，约0.6%；Other Head的移除没有解决候选／索引、状态检查、输入准备及同步成本。\n')
    table(['参数范围','四Head','去Other','差值'],[[key,params['four_head'][key],params['no_other'][key],params['no_other'][key]-params['four_head'][key]] for key in params['four_head']])
    add('移除4420 FP32参数节约17.27 KiB，Adam两阶状态34.53 KiB，梯度17.27 KiB，合计约69.06 KiB；另减少Other的batch×64中间激活及反传临时量，14D兼容输出仍保留。没有引入缓存或额外私有Feature。\n')
    timing=load('evaluation/reports/TRAINING_TIMING.json')
    table(['组','记录步数','Mapping记录均值ms','median ms','p90 ms'],[[r['group'],r['steps'],f'{r["mapping_ms_mean"]:.3f}',f'{r["mapping_ms_median"]:.3f}',f'{r["mapping_ms_p90"]:.3f}'] for r in timing])
    add('历史四Head与本轮训练不处于相同GPU负载；这些训练耗时仅为运行记录，不能据此认定完整训练加速。当前可靠的训练计算对照为共同输入前向＋反向微测。GPU驻留进程、利用率、温度与频率记录在GPU_OBSERVATIONS.json；共享GPU条件限制绝对性能结论。\n')
    add('测速期间GPU利用率99%–100%、温度85–87℃、SM时钟约1575–1875 MHz，且存在本轮以外的GPU进程。500点三轮的总耗时减少比例在−3.65%至+1.68%之间，50%预算在−3.30%至+0.18%之间，方向不一致；本轮未证明端到端稳定加速。不能拿本轮约34 ms直接与较空闲时历史约8 ms比较。\n')
    add('## 补偿与训练归因\n')
    table(['条件','Head','XYZ/ref p50','p90','p99','跨视角max','Scale倍率p99','Scale倍率max'],[
        [case,g,f'{r["xyz_p50_mean"]:.3f}',f'{r["xyz_p90_mean"]:.3f}',f'{r["xyz_p99_mean"]:.3f}',f'{r["xyz_global_max"]:.3f}',
         f'{r["scale_p99_mean"]:.3f}',f'{r["scale_global_max"]:.3f}'] for case in ['global_0.01','global_0.05','view_0.01','view_0.05','points_500','view_0.5']
        for g,_,_ in [('four_head','',''),('no_other','','')] for r in [geometry[g,case]]])
    add('XYZ/ref是各视角分位数的均值，max为全部视角最大值；另有米制位移、Rotation、Opacity、长尾实际Alpha贡献及不可见比例。不能把大Scale或位移直接当作质量下降原因。删除Other改变了网络表达能力和共享主干收到的监督；Master仍可调整Opacity/DC，Selector与Feature也随联合Photo适应，因此终态质量差不能仅归为渲染器或Head前向计算。实际训练全局／局部K曝光在TRAINING_EXPOSURE.csv。\n')
    add('已证实的是：Full仅下降0.032 dB，而全局／局部1%下降0.647／0.763 dB，损失集中在稀疏LoD路径；局部1%的XYZ/ref p99增至原来的约3.5倍。合理解释是缺少Opacity/DC补偿后，共享几何与选点需要重新适应，但当前对照没有固定选择身份及Master，不能将长尾增大直接证明为“用移动替代颜色”的因果机制。50%下p99略减，也说明它是低ratio问题。\n')
    add('## 文件与完整性\n')
    add(f'- 新模型：`{SERVER}/runs/full_no_other/23560_shutdown/ply`。')
    add(f'- 四Head对照：`{CONTROL}/runs/full_both_li/23560_shutdown/ply`。')
    add(f'- 独立源码／运行库：`{SERVER}/source`、`runtime/lib`、`runtime/bin`；运行时maps确认加载本轮独立libgaussian_mapper。')
    add('- 保护主项目／realLOD关键源码、配置、二进制、动态库604个文件；PRODUCTION_BEFORE/AFTER及FINAL_INTEGRITY逐项核验。历史模型、CSV、相机和原Full输入均核对SHA。原默认YAML保持，未commit/push。')
    add('- 加载无Other的Residual48模型需要包含本轮C++兼容补丁；本轮推理测速使用已提交默认的PyTorch路径，未混入并行开发的新CUDA推理融合。旧frozen CUDA后端不支持该几何schema，不能把这些结果称为新CUDA后端性能。')
    add('- 汇总：SUMMARY.csv、PAIRED_SUMMARY.csv；逐帧：PER_VIEW.csv、PAIRED_PER_VIEW.csv；性能：speed/TIMINGS.csv及两组微测CSV；源码快照、构建SHA、失败验收日志、训练CSV均归档。')
    add('- 不保存渲染帧图片，只保存数值、模型及四张分析曲线。\n')
    add('![ratio质量对照](figures/NO_OTHER_RATIO_QUALITY.png)\n\n![局部固定预算](figures/NO_OTHER_FIXED_BUDGET_QUALITY.png)\n\n![速度拆分](figures/NO_OTHER_LATENCY.png)\n\n![几何长尾](figures/NO_OTHER_GEOMETRY.png)\n')
    (OUT/'NO_OTHER_HEAD_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
