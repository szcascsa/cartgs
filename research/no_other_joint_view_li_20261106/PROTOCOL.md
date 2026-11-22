# 当前双视角＋LI默认模型移除Other Head的匹配消融

用户授权重新训练无Other Head模型，并测量速度与质量变化。独立目录`/home/prmi/zwj/experiments/no_other_joint_view_li_20261106`，不覆盖生产或历史模型，不修改共享环境，不commit/push。

## 唯一结构变化

基于已提交默认26c355b：Selector19→32→32→2、Residual48累加Feature、Compensator64/64主干及XYZ／Scale／Rotation三个独立Head保持。Other Head的64→64→4完全移除，减少4420个参数；不执行该Head，不分配其梯度或Adam。为兼容14D解码，仅补四维常数零；Opacity与SH DC不经过网络补偿，沿用Master，Master仍由原联合Photo优化。

复用既有geometry_head_schema=1，扩展到Residual48，新增显式Selector.geometry_head_variant=1；默认缺省0兼容旧checkpoint。旧Anchor128及mode2合并MLP行为不改变。删除Head不消耗初始化RNG，保留其余参数初态逐bit一致。共同几何输出及共享输入／参数VJP在相同上游梯度下逐bit检查，不将新训练终态等同于旧终态。

## 匹配训练

从与both_li相同的Full11980原checkpoint重启，seed0、fresh Adam/RNG、11580步，终点23560；N13298、Anchor2790、72训练关键帧，Master可更新、拓扑固定。ratio[.01,.05,.1,.3,.5,.7]各1930步；每步同相机Full→全局LoD→局部LoD逐路反传后一次Adam。LI=.1、全局GI刷新、视角编码、预算损失、学习率、Softplus Scale及原健康检查保持。共同参数、Master和完整相机／ratio序列逐值对照。无Other额外schema由原Full入口明确创建，忽略被删除Head的初始参数，不将训练过的四Head终态直接删除后冒充重训。

先独立编译全部Mapper C++对象，再单元／保存恢复／跨语言／LI隔离及36步smoke验收；正式训练重新读取原Full，不接smoke。故障必须定位，只有有证据的OOM允许额外2次重试，不盲目重训。

## 质量

792个相同RGB帧，Bank72/nonBank720，640×480、FP32、TF32关闭。Full；global／view各11ratio[.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]；局部固定500/1k/2k/3k/4k/5k/10k；五档global实际Alpha数匹配本模型局部请求预算。旧both_li已有34条件26928条只读复用，只补测旧500点的PSNR/SSIM/LPIPS及新模型35条件，合计两模型55440条。不可称跨地图matched为严格same-K。

各ratioPSNR/SSIM/LPIPS/photo、实际K／Alpha、逐帧配对、XYZ/reference及米制p50/p90/p99/max、Scale/Rotation/opacity长尾分别报告。不根据nonBank选权重／终点。全局保护沿用训练档平均ΔPSNR≥−.10dB、任一档≥−.20dB、Full≥−.10dB；局部尤其1%／5%不被平均值掩盖。单场景单seed限制明确。

## 速度

用同一进程同一冻结Python源码，交错运行旧四Head和新三Head的局部500点、局部50%（K=floor(.5Nᵥ)）、各自Master Full。各64次预热后792帧三轮顺序／逆序／乱序，全部轮次保留。共同nominal候选人口及严格预算逐帧相同。

同步墙钟分开记录Selector／视角／选择、Compensator、其他准备、Renderer和总耗时，含原有finite／状态检查，不含加载、相机构建、GT／指标／显示。CUDA Event与独立Profiler辅助归因。增加共同输入、共同batch／ratio的Compensator前向微测，区分Head算子节省与训练后地图／footprint变化。禁止用原子非确定性掩盖新数值差异，不在本轮开启未验收推理融合／Graph。

记录参数和Adam内存节省，输入／生产604文件前后SHA；不保存渲染帧，曲线PNG/PDF允许。最终中文报告、CSV、图、模型路径与可回退配置独立归档。

测速前审计发现，既有Python几何消融通过每次前向int(GPU schema)读取结构标记，会新增一次同步。将不可变标记在configure/load时缓存为Python整数，C++本来已有同样的成员缓存；checkpoint buffer保留。缓存前后及原生fixture在CPU/CUDA global/local共4组逐bit验证，单独记录SHA，不改变训练或Tensor算术。
