"""从已验收的部署证据生成中文网络性能报告与图表。"""
from pathlib import Path
import csv
import json
import statistics

HERE=Path(__file__).resolve().parent
DATA=HERE/'results/reports'


def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))


def main():
    assert read('ACCEPTANCE_COMPLETE.json')['passed'] and read('FINAL_INTEGRITY.json')['passed']
    labels={'r01':'1%','r10':'10%','r70':'70%','changing_ratio':'变化ratio','changing_points':'变化点数','jumping_view':'跳转视角＋变化预算'}
    measurements=[]
    for group in ['tum','replica']:
        runs=[read(f'BENCHMARK_{group}_{i}.json')['results'] for i in [1,2]]
        for i in range(6):
            row=dict(scene=group,condition=labels[runs[0][i]['case']],N_min=runs[0][i]['N_min'],N_max=runs[0][i]['N_max'],
                K_min=runs[0][i]['K_min'],K_max=runs[0][i]['K_max'])
            for tag in ['v11','optimized']:
                for stat in ['median_ms','p90_ms','p99_ms']:row[tag+'_'+stat]=statistics.mean(d[i]['summary'][tag][stat] for d in runs)
                for j in range(2):row[f'{tag}_run{j+1}_median_ms']=runs[j][i]['summary'][tag]['median_ms']
            row['reduction_percent']=100*(1-row['optimized_median_ms']/row['v11_median_ms']);measurements.append(row)
    with (HERE/'NETWORK_PREPARE_BENCHMARK.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,group in zip(axes,['tum','replica']):
        rr=[r for r in measurements if r['scene']==group];x=np.arange(len(rr))
        for off,tag,title,color in [(-.18,'v11','v11','#64748b'),(.18,'optimized','寄存器复用 v13','#0891b2')]:
            bars=ax.bar(x+off,[r[tag+'_median_ms'] for r in rr],width=.34,color=color,label=title)
            for bar in bars:ax.annotate(f'{bar.get_height():.3f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(x,[r['condition'].replace('＋','\n') for r in rr],fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
        ax.set_title('TUM fr1_xyz · 13,298点' if group=='tum' else 'Replica office3 · 49,102点');ax.set_ylabel('完整准备墙钟 / ms')
    axes[0].legend();fig.suptitle('逐视角重算：只测网络与准备，不含 Rasterizer',fontsize=15)
    fig.text(.5,.02,'RTX3090；两轮配对中位数的平均；存在共享 GPU 负载。每轮与长尾见 CSV。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(HERE/'NETWORK_PREPARE_SPEED.png',dpi=180);plt.close(fig)
    component=read('COMPONENT_PROFILE_ANALYSIS.json')['summary']
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,stage in zip(axes,['selector','comp']):
        names=[];old=[];new=[]
        for group in ['both_li','replica']:
            for ratio in [.01,.1,.7]:
                a=next(v for v in component if v['label']==f'node/{group}/{ratio}/{stage}_v11')
                b=next(v for v in component if v['label']==f'node/{group}/{ratio}/{stage}_optimized')
                names.append(('TUM' if group=='both_li' else 'Replica')+'\n'+f'{100*ratio:g}%');old.append(a['kernel_us']);new.append(b['kernel_us'])
                rows.append(dict(scene=group,ratio=ratio,module=stage,v11_us=a['kernel_us'],optimized_us=b['kernel_us'],
                    speedup=a['kernel_us']/b['kernel_us'],v11_kernels=a['kernels'],optimized_kernels=b['kernels']))
        x=np.arange(len(names))
        ax.bar(x-.18,old,width=.34,color='#64748b',label='v11');ax.bar(x+.18,new,width=.34,color='#0891b2',label='v13')
        ax.set_xticks(x,names,fontsize=9);ax.set_title('Selector' if stage=='selector' else 'Compensator');ax.set_ylabel('累计 GPU Kernel 时间 / μs');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle('Nsight 独立网络测量：相同输入、相同权重、相同选点',fontsize=15)
    fig.text(.5,.015,'每条件8个保存视角的中位数；不含视锥、Top-K、Rasterizer 或主机同步等待。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(HERE/'NETWORK_COMPONENT_SPEED.png',dpi=180);plt.close(fig)
    with (HERE/'NETWORK_COMPONENT_BENCHMARK.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    identity=read('BINARY_IDENTITY.json');validation=read('VALIDATION.json');integrity=read('FINAL_INTEGRITY.json')
    text=['# Selector与Compensator部署推理优化报告','',
        '日期：2026-11-07。最终实现v13，对照为已验收v11。根据用户追加要求，本轮没有修改Gaussian渲染；所有加速结果只统计Selector、Compensator或网络准备管线，Rasterizer不进入计时区。渲染只用于正确性检查。','',
        '## 1. 结论与范围','',
        '保留两项优化：Selector的16点寄存器Tile，以及高预算Compensator的16点寄存器Tile。二者均让一次权重读取供四个Gaussian的独立FMA链复用；每个输出仍按v11的输入维度顺序累加，没有分段归约、半精度、TF32、参数变化或训练变化。','',
        '没有改变名义Anchor候选、资格Mask、严格K、固定Gumbel噪声、同分优先级、最终点序、ratio编码、Residual累加、四个独立Head或属性变换。Radix、Gaussian→Tile、排序和像素合成源码均保持v11版本。','',
        'v13在本轮验收输入下相对v11的选点、属性与像素逐位一致。该结论不扩大为所有任意浮点输入的形式证明，也不扩大为部署专用网络与正常训练PyTorch版本逐位相同；后者的既有细小差异仍按之前报告理解。','',
        '## 2. 两网络实际改动','',
        '|模块|v11|v13|保持的语义|','|---|---|---|---|',
        '|Selector|每Gaussian一个Warp，Block处理4点|Block处理16点；线程拥有一个神经元的四点累加器，权重只读取一次供四点使用|11D/19D输入、32/32隐藏层、两logit、FP32 FMA序、噪声及score计算|',
        '|高预算Compensator|Block处理4点，逐层把权重装入共享内存|Block处理16点，线程将同一权重复用于四个独立点，减少权重读取和逐层装载|51D物理／Residual输入＋3D ratio投影、64/64主干、四Head、Scale／Rotation解码|',
        '|低预算Compensator|每Gaussian一个Block、四Warp并行Head|保留原实现|低预算不强行使用较慢Tile|','',
        'Selector在16点Tile内使用四个独立累加器，最后仍分别计算每点的两logit。Compensator沿Base→活跃Residual顺序进行乘法和加法；主干后依次计算四个Head，每Head仍使用原权重及偏置，不是重新训练一个合并网络。','',
        'Compensator保留1024门限：Device路径按输出容量、Host路径按实际K分流。低于门限仍走v11单点Block。相机每次改变都会重新计算，不缓存视锥Mask、选点、ratio Feature或中间网络输出。','',
        '## 3. Nsight独立网络测量','',
        '同一checkpoint、相同输入与选点；每条件8个保存相机样本。下表为各调用范围内累计GPU Kernel时间的中位数。独立包装函数的状态清零Kernel计入对应网络；不含Host等待，也不含视锥、Top-K和Rasterizer。','',
        '|场景／ratio|模块|v11 μs|v13 μs|速度倍数|','|---|---|---:|---:|---:|']
    for v in rows:text.append(f'|{v["scene"]}／{v["ratio"]}|{v["module"]}|{v["v11_us"]:.2f}|{v["optimized_us"]:.2f}|{v["speedup"]:.2f}×|')
    text+=['','![独立网络性能](NETWORK_COMPONENT_SPEED.png)','',
        '## 4. 真实变化视角下完整准备耗时','',
        '完整准备包括当前相机的候选构建、动态N/K、视角编码、Selector、稳定精确选点、Compensator及末尾状态读回。计时函数只调用prepare，不调用Renderer。TUM每轮每条件792请求；Replica每轮每条件411请求，覆盖137个保存相机三遍。A/B顺序交替，表中为两轮中位数的平均。','',
        '|场景／请求|N范围|K范围|v11 ms|v13 ms|耗时降低|','|---|---:|---:|---:|---:|---:|']
    for v in measurements:text.append(f'|{v["scene"]}／{v["condition"]}|{v["N_min"]}–{v["N_max"]}|{v["K_min"]}–{v["K_max"]}|{v["v11_median_ms"]:.4f}|{v["optimized_median_ms"]:.4f}|{v["reduction_percent"]:.1f}%|')
    text+=['','![完整准备性能](NETWORK_PREPARE_SPEED.png)','',
        '速度来自RTX3090，期间存在其他GPU进程，没有终止无关进程或锁频。每轮中位、p90、p99与逐请求原值保存在CSV／JSON。不是在线SLAM训练耗时，也不包含Rasterizer、图像编码、显示或网络传输；尚不能代表其他端侧设备。','',
        '## 5. 完整准备的GPU归因','',
        '|场景／ratio／实现|Kernel发射数|累计GPU μs|准备同步数|','|---|---:|---:|---:|']
    for v in read('PROFILE_ANALYSIS.json')['summary']:
        if v['label'].startswith('prepare/') and v['label'].endswith(('0.01','0.7')):text.append(f'|{v["label"].removeprefix("prepare/")}|{v["kernels"]:g}|{v["kernel_us"]:.2f}|{v["syncs"]:g}|')
    text+=['','网络本来已各融合为单Kernel，本轮主要减少重复权重读取、改善独立FMA调度与资源占用，所以不应把Kernel数不变理解为没有优化。准备同步次数和GPU动态N/K路径沿用v11。','',
        '编译资源统计：高预算Compensator由v11四点Tile的92寄存器／线程、20496字节共享内存／Block，变为56寄存器、16704字节共享内存；Block线程数从128到256，每Block点数从4到16。Selector新19D Tile为48寄存器、5440字节共享内存，原为46寄存器、1024字节；这些是编译资源数据，不是独立的加速证明。','',
        'Nsight Systems两份采集均完成。另尝试Nsight Compute的LaunchStats／Occupancy，但驱动返回ERR_NVGPUCTRPERM，未获得硬件计数器数据；没有修改驱动权限、锁定时钟或清空其他任务的缓存。NCU_STATUS.json和logs/ncu.log保留该限制，不能把编译资源数据当作实测占用率。','',
        '## 6. 独立开关消融','',
        '各条件120个变化相机，五种实现轮转执行，只测prepare。off使用selector_tiled=False、compensation_tile=4，恢复v11网络执行方式。','',
        '|场景／ratio|v11 ms|off|仅Selector|仅Compensator|两项开启|','|---|---:|---:|---:|---:|---:|']
    for v in read('ABLATION.json')['results']:
        d=v['median_ms'];text.append(f'|{v["group"]}／{v["ratio"]}|{d["v11"]:.4f}|{d["off"]:.4f}|{d["selector_only"]:.4f}|{d["comp_only"]:.4f}|{d["both"]:.4f}|')
    text+=['','这组消融的绝对耗时明显高于两轮主测，且TUM 1%出现非单调结果：off甚至快于v11，两项开启反而较慢。它不能单独证明各开关的墙钟收益；模块归因以相同输入的独立Nsight测量为主，实际准备收益以两轮A/B主测为主，不从消融中挑选最好的数字。','',
        '## 7. 数值、边界与异常验收','',
        f'- 五类checkpoint整体验收：{validation["cases"]}组、{validation["comparisons"]}项逐位比较，通过view模式、补偿开关、空预算、动态K、非默认Stream和返回结果所有权检查。',
        f'- 独立Selector组件及尾部尺寸：{read("SELECTOR_VALIDATION.json")["comparisons"]}项逐位比较；Compensator组件：{read("REGISTER_VALIDATION.json")["comparisons"]}项。',
        f'- 随机／极端幅度输入、11D/19D、0–65537候选、随机行序、不同Feature幅度：{read("SYNTHETIC_VALIDATION.json")["comparisons"]}项逐位比较通过，异常状态同样比较。']
    for group in ['tum','replica']:
        d=read(f'TRAJECTORY_VALIDATION_{group}.json');text.append(f'- {group}：{d["frames"]}相机×18条件={d["cases"]}请求，{d["comparisons"]}项比较，ID／属性／像素／radii逐位一致。')
        d=read(f'CONTROL_AUDIT_{group}.json');text.append(f'- {group}随机ratio／预算：{d["cases"]}请求，ID差异{d["ids_mismatches"]}、属性差异{d["attribute_mismatches"]}；CPU/GPU控制数学的既有末位差异仍单独记录。')
    text+=['- Selector／Compensation非有限值检查、错误后恢复、源模型修改不影响冻结实例、源模型与RNG不变均通过；两场景Compute Sanitizer memcheck为0 errors。',
        '- 重新加载正式checkpoint及独立关闭开关36组通过；98204点合成容量的Host保留路径39组通过，后者不是新大场景速度证据。',
        '- 最终渲染像素逐位相同，因此在相同参考图下PSNR/SSIM/LPIPS不会变化；没有用PSNR接近替代数值验收，没有保存渲染图片。','',
        '## 8. 显存与状态生命周期','',
        '|场景／实现|冻结实例保留MiB|100次变化请求峰值增量MiB|','|---|---:|---:|']
    for group in ['tum','replica']:
        for tag in ['v11','optimized']:
            d=read(f'MEMORY_{group}_{tag}.json');text.append(f'|{group}／{tag}|{d["retained_bytes"]/1024**2:.3f}|{d["peak_extra_bytes"]/1024**2:.3f}|')
    text+=['','没有新增持久的网络中间Tensor或结果缓存，新增Tile使用Block内临时共享内存和寄存器。冻结实例、UID、拓扑和有界Workspace生命周期沿用v11；更换checkpoint或地图应重建实例，同一实例串行调用。','',
        '## 9. 其他候选与舍弃原因','',
        '|候选|实测|处置|','|---|---|',
        '|批量FP32 GEMM＋四Head隐藏层合并调度|216组初步质量检查通过；高预算组件约快15%–20%，低预算更慢；与v11有属性末位差和局部像素差|未纳入最终版本；寄存器候选更快且逐位一致|',
        '|32点寄存器Compensator Tile|与16点同样通过原组件逐位检查，但多数代表条件更慢，中小K尤其明显|未纳入最终路径，保留原阶段证据|',
        '|低预算强行使用16点Tile|约84／217点时明显慢于v11单点Block|保留低预算旧Kernel与1024门限|',
        '|Gaussian→Tile／Rasterizer优化|用户明确本轮不做|只读核查，没有实现、编译或替换渲染器|','',
        'GEMM原始测量和失败入口导入现场留在GEMM_TEST.json与logs；导入错误是研究目录缺少当前模型加载依赖，已复制只读v11依赖修复，没有修改checkpoint。另一次GEMM解码改写未做性能验证，没有据此声称收益。','',
        '## 10. 接入、回退与完整性','',
        '新增部署参数：selector_tiled=True；compensation_tile默认16。独立回退本轮网络优化：','',
        '```python','runtime = DeploymentRenderer.from_checkpoint(','    checkpoint, selector_tiled=False, compensation_tile=4)',
        '```','',
        '正常训练、训练中的推理、常规评价和Viewer默认后端均未切换。服务器仅写独立目录：/home/prmi/zwj/experiments/deployment_network_gemm_20261107。没有commit/push或生产部署。','',
        f'最终二进制：{identity["path"]}；SHA256={identity["sha256"]}。基准v11 SHA256={identity["oracle_sha256"]}。逐文件源码身份见BINARY_IDENTITY.json，本轮源码未提交。','',
        f'使用原Rasterizer二进制，SHA256={identity["rasterizer_sha256"]}，前后未变化。','']
    for n,g in integrity['groups'].items():text.append(f'- {n}：{g["files"]}文件，修改{len(g["changed"])}，缺失{len(g["missing"])}。')
    text+=['','本地修改仅涉及部署网络、调度、绑定和部署说明；cuda_rasterizer及正常训练文件相对本轮开始时均未变化。原始配置和模型保留，未清理无关文件或结束无关进程。','',
        '## 11. 剩余网络优化空间','',
        '1. 低预算现在主要剩候选准备、精确选择及Host调度；可研究固定地址、有界Workspace和CUDA Graph，每帧更新相机／ratio／预算，不缓存计算结果。',
        '2. 1024阈值仍按容量或Host K判断，并非每个视角的最佳分流。可用GPU实际K研究更细的分流，但不能为此增加中途读回或牺牲小预算速度。',
        '3. 更大真实地图和目标端侧GPU尚未测量；这里的RTX3090结果不能直接作为端侧性能保证。应在目标设备重新测寄存器限制、Tile大小及尾延迟。',
        '4. 高斯渲染优化继续留待用户另行指示；本轮所有网络收益均未计入任何渲染器提速。','']
    (HERE/'NETWORK_INFERENCE_OPTIMIZATION_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print('NETWORK_REPORT_READY',len(measurements))


if __name__=='__main__':main()
