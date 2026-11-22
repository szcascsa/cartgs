# Other Head冻结推理贡献消融

2026-11-06，用户要求选择既有实验，关闭Other Head补偿后渲染，检查其实际贡献。本轮选择已完成的fr1_xyz RGB-D、两网络真实视角＋LI=0.1模型，seed0、Fine终点23560，N13298、Anchor2790。

不重训。Other Head为Opacity logit1＋SH DC3。每帧先准备原选点与所有补偿属性，再复制渲染输入字典，仅将Opacity和DC替换为同一Master对应行的原值。XYZ、Scale、Rotation、SH其余项和全部保留点K不变。不删除模型权重、不改变槽位准入或重新选点。

两个条件同进程配对重渲染，避免只用历史值推断贡献。首／中／末三帧×28条件将网络输出的最后4维直接归零，检查与替换渲染属性方式逐bit一致；另做168次普通／计数Renderer颜色及radii逐bit检查。所有22176组选择身份及K与既有实验一致；模型state_dict前后逐bit核验。

评价792 RGB帧，Bank72、nonBank720。global／local各11ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]，局部固定1k／2k／3k／4k／5k／10k，共28条件。局部真实视角、名义Anchor候选池；全局零视角条件、全图候选。局部严格K与全局分位数沿用原推理口径，候选不足则保留实际K。固定预算的网络ratio可能外推，单独报告。

FP32、TF32关闭，颜色clamp[0,1]；逐帧PSNR平均，TorchMetrics11×11 SSIM、AlexNet LPIPS(normalize=True)、L1，batch8。记录每帧选点SHA、实际K、radii点数、实际Alpha贡献点数、Opacity／DC补偿幅度。报告Other保留−关闭的配对mean／median／p10／p90及改善帧比例，正PSNR／SSIM、负LPIPS表示保留有帮助。

完整Full不使用Compensator，不受本消融影响，引用同一checkpoint既有Full指标，不重复Full渲染。关闭Other后Alpha贡献点数可能变化，不强行匹配该数量，否则会改变选点集合。结论仅为既有联合训练模型对Other的推理依赖，不能当成删除Other后重训的能力上限，也不能在PSNR dB上解释成可加的贡献百分比。

独立目录`/home/prmi/zwj/experiments/other_head_inference_20261106`，Python源码从已完成实验快照独立复制。模型、数据和第三方CUDA依赖只读复用。不使用当前工作区其他任务的未提交推理改动，不修改生产、共享conda或历史实验，不commit／push，不保存渲染帧图片。共享GPU负载下墙钟只记成本，不称推理加速。前后核验604个生产文件和所有模型／历史输入哈希。
