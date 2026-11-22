# 固定预算网页 Viewer

当前默认入口为网页工作台，默认启用融合 CUDA 推理。原生 CaRtGS 固定预算 Viewer 通过 `--viewer native` 保留。此切换只影响可选推理入口，不改变 SLAM 训练默认参数、网络、点数预算或 checkpoint。

## 启动

服务器 `/home/prmi/zwj/realLOD`：

```bash
/home/prmi/anaconda3/envs/zwj/bin/python /home/prmi/zwj/realLOD/scripts/view_budget.py \
  /path/to/shutdown \
  --dataset /path/to/dataset --budget 5000 --port 8765
```

本地建立转发，再访问 `http://127.0.0.1:8765`：

```bash
ssh -N -L 8765:127.0.0.1:8765 prmi@172.19.163.58
```

省略 `--port` 时选择空闲本机端口，并打印实际地址与对应转发命令。网页模式不需要 DISPLAY 或图形会话；服务仅监听回环地址。启动器退出时只回收自己启动的服务，不影响其他 Viewer 或训练进程。

推理后端：

- 默认 `--inference-backend cuda`：现有 FP32 融合实现，保持原 GEMM 与模型定义。
- `--inference-backend cuda_graph`：使用既有有界形状缓存；动态图形捕获限制沿用原实现。
- `--inference-backend pytorch`：显式选择原 PyTorch 路径。

不支持融合布局的 checkpoint 会明确显示 PyTorch 兼容路径及原因。CUDA 编译失败、非有限值或渲染运行异常会直接报错，不把错误伪装成正常结果。实际后端同时记录在元数据及逐帧指标中。

保留原生客户端：

```bash
/home/prmi/anaconda3/envs/zwj/bin/python /home/prmi/zwj/realLOD/scripts/view_budget.py \
  /path/to/shutdown --dataset /path/to/dataset --viewer native --display localhost:12.0
```

DISPLAY 取当前实际 X11 会话。原生 FP32 图像协议仍可使用。

## 工作台布局

左侧为操作控制，中间为渲染画面、关键指标与轨迹，右侧为概览／性能／分组三个面板。窄屏时统计移至画面下方，手机视口优先显示画面。相机灵敏度、显示与诊断默认折叠，避免挤占预算与消融操作。

| 模式 | 候选与选点 |
|---|---|
| 可见预算 | 启用层级中、名义格子与视锥相交的 Anchor 成员；支持点数或比例预算。 |
| 全图训练 ratio | 不按当前视锥裁剪；保持训练对应的全图 FlexGS 推理规则，之后隐藏关闭的层级。 |
| Full Master | 启用层级的原始未补偿成员，跳过两网络。 |
| Anchor 分档 | 可见成员按投影尺寸或相机深度排序、按 Gaussian 数累计分组；支持三档与既有 mixed ratio。 |

点数预算 B 以 `r=min(1,B/N)` 为网络条件；比例预算直接使用请求 r，`B=floor(rN)`。预算模式严格 `K=min(B,eligible_count)`，不通过改变训练或随机规则补齐。全图模式保留原分位数规则与同分行为，界面显示实际 K。

四组消融：A 学习选择＋补偿、B 学习选择不补偿、C 随机选择＋补偿、D 随机选择不补偿。随机组沿用固定随机键。带视角条件的 checkpoint 在支持的推理路径显示对应开关。

Anchor 0／1／2 为真实 `anchors.levels`，三个独立复选框支持任意组合；与空间 ratio 高／中／低分档分别统计。全部关闭渲染背景。原有筛选、严格 K、原始成员顺序、缓存失效检查与 Feature 语义保持不变。

## 相机、轨迹与对照

- 左键拖动旋转、右键拖动滚转、中键拖动平移、滚轮前后移动。
- 点击画面后，持续按 WASD 移动，Q/E 上下移动，I/K 俯仰、J/L 偏航、U/O 滚转；Shift 加速，R 复位，F 全屏。输入框保留正常编辑行为，失焦清除按键。
- 可调移动、旋转与三种鼠标灵敏度；支持适应窗口或 1:1 像素显示，不改变服务端分辨率。
- 支持上一帧、下一帧、时间轴、相机列表、帧号跳转、复位、刷新与播放；播放上限 2／5／10 帧每秒。每帧完成后再推进轨迹，避免慢服务下跳过大量相机或堆积请求。切到后台或窗口失焦暂停播放。
- 参考对照按钮仅改变布局；数据集相机显示精确配对 RGB，自由视角不使用最近RGB冒充真值。
- Full 对照 PSNR 需勾选后额外渲染，同视角、同启用层级。分档着色也会额外渲染，PSNR仍来自正常外观。

默认 JPEG 95 传输；可切换 PNG 无损 8-bit 显示。两者均使用原分辨率，PSNR 在图像编码前的 FP32 结果上计算。原生 GL 纹理上传／交换耗时与网页显示方式不同，网页对应提供 HTTP、解码与请求到画面更新时间，不伪造 GL 指标。

保存画面导出已显示帧的实际传输图；导出指标保存该帧的请求设置、实际相机、服务端统计及浏览器耗时，避免待处理的新设置与旧画面混配。

## 开销与缓存

浏览器最多一个在途请求；拖动或连续按键时只合并最新状态，显示最近完成的真实帧。空闲时停止请求。帧下方提示是否仍有新操作待渲染。

浏览器保存一张当前参考图。相同视角、层级和图像格式时发送参考缓存身份，服务端跳过相同参考图的重复读回、编码、传输与解码。切换相机、参考范围或格式时更新；不改变PSNR计算。

冻结 Master、固定随机键、最多8张RGB真值、最多8项全图结果与原有 CUDA Graph 缓存沿用原实现。任何参数或拓扑更新仍需重新加载／rebuild，不新增无界输出缓存。

性能面板将准备、光栅化、Anchor统计、质量与参考、图像编码、HTTP往返、浏览器解码分别列出。Selector、Compensator、分组包含在准备中，Full光栅化包含在质量与参考中，不能重复相加。准备＋光栅化不等于端到端帧时间。

## 范围与交付

本轮迁移对象是冻结 checkpoint 上的原生固定预算 Viewer。实时 SLAM 的训练参数滑块、在线 Tracking 控制和稀疏地图绘制需要在线 Mapper／ORB 进程接口，不在只读 checkpoint 服务中放置无效控件。

代码：`scripts/view_budget.py`、`python/budget_viewer.py`、`python/budget_viewer.html`、`python/budget_viewer.css`、`python/budget_viewer.js`。未改神经网络或核心 CUDA Kernel，无需重新编译 SLAM。

本轮仅完成语法／静态检查及无模型的界面预览，按此前要求未运行训练、质量评估、CUDA性能实验或新增测试。历史质量与性能数字不作为本轮结果。部署及备份证据见 `/home/prmi/zwj/experiments/web_viewer_primary_20261105/`；更新后重新启动新的 Viewer 服务，已有独立实例继续使用它们原来的源码。
