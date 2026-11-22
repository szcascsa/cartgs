"""局部ratio=0.5的同步延迟、真实点数和质量报告。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'reports';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
S={(v['method'],v['repeat']):v for v in json.loads((OUT/'SUMMARY.json').read_text())}
Q={v['method']:v for v in json.loads((OUT/'QUALITY_SUMMARY.json').read_text())}
P=json.loads((OUT/'PROFILE_SUMMARY.json').read_text())
events=json.loads((OUT/'CUDA_EVENT_SUMMARY.json').read_text())
observations=json.loads((OUT/'GPU_OBSERVATIONS.json').read_text())
pid=str(json.loads((ROOT/'STATUS.json').read_text())['pid'])
names={'cartgs_full':'原版CaRtGS Full','cartgs_full_cached':'原版Full（静态属性缓存）',
       'default_500':'当前默认局部500点','default_ratio50':'当前默认局部ratio=0.5','default_full':'当前默认Master Full'}
methods=list(names)
local=S['default_ratio50','all'];fixed=S['default_500','all'];full=S['cartgs_full','all'];q=Q['default_ratio50']
lines=['# 当前视角ratio=0.5的推理与渲染速度','',
    f"**本轮局部50%平均实际保留{q['K']['mean']:.2f}个Gaussian（{q['K']['minimum']:.0f}–{q['K']['maximum']:.0f}），推理／准备{local['inference_ms']['mean']:.3f}ms，渲染{local['render_ms']['mean']:.3f}ms，总计{local['total_ms']['mean']:.3f}ms。**",'',
    f"同轮原版CaRtGS Full总计{full['total_ms']['mean']:.3f}ms，局部50%耗时为其{local['total_ms']['mean']/full['total_ms']['mean']:.2f}倍。同一默认模型局部500点总计{fixed['total_ms']['mean']:.3f}ms，50%相对500点变化{100*(local['total_ms']['mean']/fixed['total_ms']['mean']-1):+.2f}%。",'',
    '这是已提交默认26c355b的PyTorch推理路径，两网络真实视角＋LI=0.1训练所得终态，未训练、未修改生产配置。GPU负载状态按轮次列出，不与上次不同时段的500点数字直接作因果比较。', '',
    '## 预算与真实计算范围','',
    'Nᵥ为名义Anchor格子与当前视锥相交后、所包含的全部Gaussian成员数。请求K=floor(0.5Nᵥ)，实际K=min(请求K,合格成员数E)。这不是全局地图的一半，也不是实际Alpha可见点数的一半。Selector和Compensator均使用真实view7＋mode1，网络ratio固定0.5；全局导出仍用零条件。', '',
    f"当前地图13298点、2790 Anchor；候选Nᵥ平均{q['candidate_N']['mean']:.2f}，范围{q['candidate_N']['minimum']:.0f}–{q['candidate_N']['maximum']:.0f}。候选不足帧数{q['budget_shortfall_frames']}。所有保留点均计入K，包括不贡献像素的点。Selector仍逐候选点评分，补偿和Renderer处理实际K个点。",'',
    '## 测量协议','',
    'TUM fr1_xyz RGB-D、RTX3090、640×480、FP32、TF32关闭、batch1。26个Python文件直接从26c355b的Git对象提取，未混入并行工作区中的推理优化。与上次相同原版Full checkpoint（35301点）及当前默认checkpoint；相同792个RGB时间戳，原版使用自己的SLAM位姿，500点／50%／Master Full共享本方法完全相同的位姿。', '',
    '五方法各64次预热；完整792帧测三轮，帧次序顺序／逆序／seed0乱序，方法次序逐帧轮转，共11880次主计时，每方法2376次。每次局部重新执行候选构建、Selector、选点、Feature／Compensator和打包，不缓存最终选点／补偿。', '',
    '推理／准备包含名义视锥、Gather、资格／预算、view编码、Selector、Gumbel／稳定排序、补偿、finite与状态检查。渲染计时包含Renderer调用和等待GPU完成。模型加载、相机准备、GT、指标、Viewer传输显示均在计时之外。原版标准入口含它原有的属性激活和screenspace准备；缓存对照预计算静态Master属性，仍执行同一原版CUDA渲染，首／中／末颜色和radii逐bit一致。', '',
    '|方法|实际K均值|推理准备ms|渲染ms|总均值ms|总median ms|总p90 ms|','|---|---:|---:|---:|---:|---:|---:|']
for m in methods:
    v=S[m,'all'];lines.append('|'+names[m]+'|'+f"{v['K']['mean']:.2f}|"+'|'.join(f"{v[k]['mean']:.3f}" for k in ['inference_ms','render_ms','total_ms'])+f"|{v['total_ms']['median']:.3f}|{v['total_ms']['p90']:.3f}|")
lines+=['','|方法|首轮总ms|第二轮总ms|第三轮总ms|','|---|---:|---:|---:|']
for m in methods:lines.append('|'+names[m]+'|'+'|'.join(f"{S[m,str(k)]['total_ms']['mean']:.3f}" for k in range(3))+'|')
lines+=['','|轮次|开始时其他GPU进程|结束时其他GPU进程|','|---|---|---|']
for k in range(3):
    def other(label):
        value=next(v for v in observations if v['label']==label)
        return ','.join(line.split(',')[0].strip() for line in value['processes'].splitlines()
                        if line.split(',')[0].strip() not in [pid,'1027483','2502783','2502784']) or '无（除原三个Viewer）'
    lines.append(f"|{k+1}|{other(f'round_{k}_start')}|{other(f'round_{k}_end')}|")
lines+=['','边界快照不能证明块内部没有短任务，所有轮次保留，不按速度筛除。GPU时钟、温度、利用率及进程显存见GPU_OBSERVATIONS.json。不同地图和不同渲染实现的比较仅代表本次实际入口的端到端调用速度，不代表同输入的CUDA算子加速，也不代表完整SLAM／Viewer FPS。','',
    '## 同一模型500点与50%分解','',
    '|预算|Selector／view／选点ms|Compensator ms|其他准备ms|渲染ms|总ms|','|---|---:|---:|---:|---:|---:|']
for m in ['default_500','default_ratio50']:
    v=S[m,'all'];lines.append('|'+names[m]+'|'+'|'.join(f"{v[k]['mean']:.3f}" for k in ['selector_ms','compensation_ms','other_prepare_ms','render_ms','total_ms'])+'|')
lines+=['','50%预算不会减少Selector候选数量；它会提高被补偿与提交渲染的Gaussian数，并使Residual48查询启用ratio=0.5对应的残差级别。计时还受Kernel形状、GEMM吞吐、可见footprint和调度影响，因此不按点数线性推算耗时。Selector栏包含view编码／概率／排序，不是纯MLP时间。','',
    '## 图像质量与点数','',
    '|方法|PSNR dB|实际K|radii>0均值|','|---|---:|---:|---:|']
for m in ['cartgs_full','default_500','default_ratio50','default_full']:
    v=Q[m];lines.append('|'+names[m]+f"|{v['psnr']['mean']:.4f}|{v['K']['mean']:.2f}|{v['projected_K']['mean']:.2f}|")
lines+=['','质量另作全轨迹792帧评价，不计入速度。radii>0不等于Alpha贡献点数。本轮不以更低画质冒充等质量加速；500点与50%质量和点数都明确列出。首／中／末的重复推理选点与颜色逐bit一致，所有帧的严格预算均通过检查。','',
    '## CUDA Event与Profiler','',
    'CUDA Event均匀96帧×五方法，保留stream内CPU发射空隙／内部同步影响，不把它当纯Kernel累计时间。独立Profiler四帧×五方法，只对实际kernel事件求和；Profiler墙钟不用于主测速。','',
    '|方法|Kernel Launch均值|Kernel累计ms|Device同步次数|Stream同步次数|','|---|---:|---:|---:|---:|']
for v in P['summary']:lines.append('|'+names[v['method']]+'|'+'|'.join(f"{v[k]['mean']:.3f}" if k=='kernel_ms' else f"{v[k]['mean']:.1f}" for k in ['kernel_launches','kernel_ms','device_syncs','stream_syncs'])+'|')
lines+=['','同步次数包括测试外层同步和原prepare内部同步，并非全部可删除。Event逐方法均值和原Profiler trace随包交付。不能直接用某个Kernel时间替代实际动态推理总延迟。','',
    '## 完整性与交付','',
    '服务器独立目录：`/home/prmi/zwj/experiments/fr1xyz_local50_latency_20261106`。生产604关键文件、模型、轨迹、扩展及本轮源码前后SHA通过；不训练、不覆盖旧结果、不修改生产、不commit/push，不保存渲染图片。', '',
    '主文件：TIMINGS.csv、SUMMARY.csv、PAIRED_COMPARISONS.json、QUALITY_COUNTS.csv、QUALITY_SUMMARY.json、CUDA_EVENTS.csv、PROFILE_SUMMARY.json、INPUTS_BEFORE/AFTER.json、FINAL_INTEGRITY.json。单场景单设备，不外推至所有地图与分辨率。','',
    '![50%预算速度](figures/LOCAL50_LATENCY.png)']
(OUT/'LOCAL50_RENDER_LATENCY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
short=['CaRtGS Full\n35,301点','CaRtGS Full\n静态属性缓存','当前默认\n局部500点','当前默认\n局部50%','当前默认\nMaster Full']
fig,axes=plt.subplots(1,2,figsize=(14,5.6));x=np.arange(5)
infer=np.array([S[m,'all']['inference_ms']['mean'] for m in methods]);render=np.array([S[m,'all']['render_ms']['mean'] for m in methods])
axes[0].bar(x,infer,label='推理与准备',color='#0072B2');axes[0].bar(x,render,bottom=infer,label='渲染',color='#E69F00')
for i,v in enumerate(infer+render):axes[0].text(i,v+.1,f'{v:.3f}',ha='center',fontsize=9)
axes[0].set_xticks(x,short,fontsize=8);axes[0].set(ylabel='三轮平均每帧耗时（ms）',title='动态推理＋渲染',ylim=(0,(infer+render).max()*1.18));axes[0].legend(frameon=False)
qm=['cartgs_full','default_500','default_ratio50','default_full'];values=[Q[m]['psnr']['mean'] for m in qm]
bars=axes[1].bar(np.arange(4),values,color=['#64748B','#009E73','#0072B2','#E69F00'])
axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9);axes[1].set_xticks(np.arange(4),['CaRtGS Full','局部500点','局部50%','Master Full'])
axes[1].set(ylabel='792帧平均PSNR（dB）',title=f"局部50%实际K均值{q['K']['mean']:.1f}点",ylim=(0,max(values)*1.12))
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('fr1_xyz RGB-D｜当前视角ratio=0.5与固定500点',fontsize=14)
fig.text(.5,.01,'RTX3090｜640×480｜相同模型与相机｜792帧×3轮交错测速｜已提交默认PyTorch路径',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ['png','pdf']:fig.savefig(FIG/('LOCAL50_LATENCY.'+ext),dpi=190,facecolor='white')
plt.close(fig)
print('中文报告与图已生成')
