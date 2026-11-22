# Neural V1复现入口

服务器172.19.163.58，实验根目录`/home/prmi/zwj/neural-v1-20261027`，Python=`/home/prmi/anaconda3/envs/zwj/bin/python`。使用现有mycartgs Renderer，不需要隔离构建。认证单独提供，不写入文件。

## 来源

`scenes.json`给出两张冻结地图和数据集。`fixture_*.json`给出具体选择身份与原补偿参照的SHA；`frozen_fixtures/`保留其只读副本。`environment.json`记录真实服务器核心源码和Renderer SHA；`research_state.json`给出本地Git快照，两者不可混用。

`dependencies/`保留历史Reference59代码链与协议快照，当前runtime仍使用原绝对路径以符合旧代码SHA约束；转移主机时需按manifest重建路径或显式记录路径修改。数据集和原Master地图不是交付包的完整副本，必须保留原始来源。纯候选本身只依赖`neural.py`，没有历史补偿表或Teacher参数依赖；训练Fixture读取R3数据是为了固定集合和独立质量参照。

## 训练

在实验目录执行。完成的运行通过`complete.json`自动跳过。新重复应复制queue到新tier，不覆盖历史结果。

```bash
python run.py --queue queue_confirm_A.json
python run.py --queue queue_confirm_B.json
python run.py --queue queue_reference_control.json
python run.py --queue queue_condition_controls.json
python run.py --queue queue_fr2.json
python run.py --queue queue_baseline_sets.json
python run.py --queue queue_long_5100.json
python run.py --queue queue_long_5101.json
python run.py --queue queue_node_clock.json
python run.py --queue queue_folded.json
python run.py --queue queue_input_cause.json
python run.py --queue queue_repeat.json
python run.py --queue queue_fr2_local.json
```

screen、transport、moment、covariance、SIREN队列保留全部短程记录。Teacher诊断用`teacher_probe.py --queue queue_teacher_corrected.json`，其后微调用`run.py --queue queue_teacher_finetune.json`。原`teacher_probe`作业因零旋转梯度错误被中止，保留无效记录，不用于算法结论。

## 评价与分析

```bash
python evaluate.py --plan plan_confirm.json
python evaluate.py --plan plan_reference_control.json
python evaluate.py --plan plan_controls.json
python evaluate.py --plan plan_fr2_confirm.json
python evaluate.py --plan plan_baseline_sets.json
python evaluate.py --plan plan_long36k.json
python evaluate.py --plan plan_node_clock.json
python evaluate.py --plan plan_folded.json
python evaluate.py --plan plan_input_cause.json
python evaluate.py --plan plan_repeat.json
python evaluate.py --plan plan_fr2_local_followup.json
python summarize.py
python final_audit.py
python plots.py
python report.py
```

每模型`formal/`保存Bank训练/Bank留出/non-Bank逐视图指标、原始输出和源SHA。最终固定12k或总36k终点，不按test挑checkpoint。`quality_per_view.csv`、`paired_results.csv`提供逐视图改善分布；`NEURAL_V1_TRAINABLE_PARAMETERS.csv`和每模型审查列出纯网络全部参数。

## 机制与性能

```bash
python diagnostics.py
python covariance_gate.py
python node_clock_gate.py
python local_units_gate.py
python mechanism_probe.py --plan mechanism_plan.json
python endpoint_coverage.py
python endpoint_support_probe.py
```

最后两项为正式评价之后的只读Bank机制诊断；不会训练、修改模型或选择checkpoint，其结果不能冒充独立视图泛化。

微基准只计当前Python补偿调用与真实photo VJP反传；不是完整在线Mapping时间。运行时仍有其他用户GPU负载；Baseline helper包括N行编码，候选只处理K行，这一范围差异已在报告注明。

## 完整性与状态

- `artifact_manifest.json`列出证据与全部正式终点SHA；`archive_integrity.json`给出两个压缩包SHA。
- `neural_v1_models.tar.gz`含57个正式终点，包括所有seed和同seed重复；不按最高non-Bank挑选模型。短程和中途checkpoint继续保留在服务器。
- 初批完成时读取源码SHA，loaded版本精确记录有缺口；后续import时保存SHA与源码快照，CSV明确区分。
- 所有正式候选通过共享Linear参数、无ID/Embedding/场景Parameter、两地图参数shape一致及相同K审查。
- 没有修改生产配置，没有stage、commit或push；没有在线Joint或C++/CUDA集成。
