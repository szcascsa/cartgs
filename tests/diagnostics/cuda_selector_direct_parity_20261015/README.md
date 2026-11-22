# Selector 直接 parity 验收

固定同一个 checkpoint、Gaussian/metadata、ratio、noise 和 upstream；克隆为 legacy A / legacy B / CUDA，独立完整 SLAM 不参与 gate。

`direct.py` 读取七个冻结快照，比较 input preparation、八层激活、logits/probability、cached uniform 和六个固定 Gumbel seed、全部输入/18 参数 VJP、共享 target 的 ratio/GI/photo/Guard/Replay/CD/DSD；每个快照再跑 1000 步 Native Adam 训练。`bindings.cpp` 直接调用生产 CUDA 网络、Native LibTorch Adam/CUDA Adam 与 AdamStepPreview，不重写优化器数学。

forward：max abs≤1e-6、relative L2≤1e-6、cos≥.999999。gradient：relative L2≤1e-5、cos≥.99999；接近零的参考 Tensor（RMS≤1e-7）可用 error RMS≤1e-7、max≤1e-5 判定。每个上界结合两倍 legacy repeat error，cosine 使用对应 error=1−cos 的规则。实际选择 hard disagreement/count difference≤.0001，至少99% mismatch 的 legacy margin≤1e-6。all-tie 仅记录诊断。

训练固定 Gaussian 输入，仅更新 Selector 的 18 参数；三个模型使用相同 frame slot/ratio/noise/targets/LR，逐步比较 output、loss、输入/参数 gradient、参数、Adam m/v/step。没有 renderer backward、拓扑变化或独立 SLAM 输入。

原始逐 Tensor 指标写 JSONL，逐 step 汇总写 CSV；实际 mismatch 的全部 global IDs 和 margin 单独保存。性能使用既有 Native benchmark 边界重新测量 100 warmup+500 次，不能将 structural caller 数字冒充 Mapping wall。
