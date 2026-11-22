# Compensation Scale的归一化Softplus解码

日期：2026-11-02。

## 定义

Scale Head的三个输出改为无量纲相对参数u，解码为：

`s = max(exp(master_log_scale) * softplus(log(expm1(1)) + u), FP32_tiny)`。

其中常数为0.5413248546129181，FP32_tiny为1.1754943508222875e-38。Softplus使用beta=1、threshold=20的稳定实现。有限输入下输出不会为负；极端负输出下的数值下限防止下溢到零，不是物理尺度上限。NaN保留，不用下限掩盖异常。

u=0时Softplus因子为1；正常尺度范围内保持原始Master尺度。u<0可缩小，u>0可放大。正方向增长近似线性。除数值下限区域外，Master log-scale梯度为上游梯度乘补偿后Scale，Head梯度为上游梯度乘Master Scale乘sigmoid(0.5413+u)。Head零初始化时仍有有效梯度。

## 配置和适用架构

```yaml
Selector.physical_scale_additive: 0
Selector.softplus_scale: 1
```

两项不能同时开启。省略softplus_scale时仍为0，历史独立配置保持原语义。关闭两项恢复原Log-Scale残差；仅physical_scale_additive=1恢复物理直接加法。

本次在当前Residual48架构上实现，没有恢复历史Anchor128结构。默认selector_config.yaml、spatial_offset3_residual48.yaml及用户指定的anchor_context_residual48.yaml启用Softplus；其余配置不自动升级。Anchor上下文Selector、Feature结构、ratio条件化、Maturity、SGI和训练调度保留各配置原值。

## 实现范围

- CUDA Decode14新增独立模式，前向和VJP保持一个Kernel完成全部属性解码。
- 非融合C++路径与Python评价使用相同公式。
- 模型新增非训练buffer `softplus_scale_schema=1`，不新增可训练参数，不消耗初始化RNG。
- 原生加载、Python完整Selector加载、Frozen teacher克隆保留该标记。
- 旧模型根据原有schema继续使用Log-Scale或物理加法。已按Softplus配置创建的原生模块拒绝直接加载不匹配的旧权重，不能把旧Head数值静默解释成新单位。
- XYZ、Rotation、Opacity和SH DC的解码未改变。Master仍保存原来的log-scale，不修改地图存储表示。

## 服务器更新范围

仅更新用户指定的隔离验证目录：

`/home/prmi/zwj/experiments/anchor_context_residual_20261102/`

源码位于source/；构建位于build/；可执行文件及动态库位于source/bin/和source/lib/。本轮证据、源文件更新前备份及构建日志单独放在softplus_update_20261102/，不覆盖历史验证报告。

服务器主项目不部署；不恢复任何训练队列；不启动完整SLAM训练；不stage、commit或push。

## 验证与限制

本轮检查零输出身份、正值缩放、极端输出、CPU/CUDA VJP、Python数值梯度、完整Selector保存加载及Frozen克隆、旧checkpoint兼容，以及真实Renderer/Joint backward/Adam的短Mapping fixture。最终结果以research/softplus_scale_20261102/verification.json及相应日志为准。

这项修改解决补偿Scale直接加法越过零的问题，不保证Master的exp激活、Renderer方差、Quaternion或位姿优化不再出现非有限值。很大的正输出仍可能产生大Scale，极负输出仍可能进入饱和/数值下限区域。本轮没有新的完整在线质量结果，不能据此宣称PSNR或整条训练链稳定性改善。
