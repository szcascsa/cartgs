# ChangeDistill 分支 epochs

沿用现有 OpenCV dotted YAML 配置：

```yaml
ChangeDistill.epochs: 5
ChangeDistill.selector_epochs: 20
ChangeDistill.tf_epochs: 5
```

`selector_epochs` 和 `tf_epochs` 分别覆盖对应分支；未设置的分支回退到
`ChangeDistill.epochs`。因此仅有旧 `epochs: 5` 的配置仍执行 S5T5。
三项 epoch 值必须为正整数，关闭 CD 仍使用 `ChangeDistill.enable: 0`。

默认配置不变。保存配置时保留旧字段，并只保存显式设置的新字段。
训练 loss、Adam、学习率、batch、采样、teacher、AABB 和 Mapping 时钟均不变。
生产代码继续使用既有随机数流程，因此改变 Selector epoch 数也会改变
随后 TF shuffle 的随机数位置。严格配对实验额外在实验 runner 中固定
TF 的随机数起点；此控制不进入生产代码。

`change_distill_test`、`change_distill_selector_cuda_test`、
`change_distill_tf_cuda_test` 包含旧 5 与显式 S5T5 的参数一致性测试，
以及非对称 S2T3 的 optimizer step 数检查。
