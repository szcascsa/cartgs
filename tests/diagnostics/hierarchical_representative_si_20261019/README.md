# Hierarchical 3-Representative SI correctness 证据

最终参数：初始总数2048 cells、3 reps/leaf、Q<0.80才split、最大L3、effective maturity=1。用户已明确2048表示总数，不表示最短边分段数。

结论为 **IMPLEMENTATION PASS**，不代表质量、online或production GO。默认开关关闭，没有commit/push。

完整报告：[HIERARCHICAL_REPRESENTATIVE_SI_IMPLEMENTATION_REPORT.md](../../../HIERARCHICAL_REPRESENTATIVE_SI_IMPLEMENTATION_REPORT.md)。

- `synthetic_cpu.log`、`synthetic_cuda.log`：CPU/GPU synthetic与legacy parity。
- `global_gi_regression.log`：原GlobalGI测试。
- `mapper_compile_check.json`：当前production参数下的mapper translation-unit编译检查；不是完整SLAM重链接/运行。
- `fr1_room_telemetry.json`、`fr1_room_tree.csv`：真实冻结map一次构建；leaf=1才产生输出，其他记录是被废弃的provisional parent。
- `independent_smoke_verification.json`：独立NumPy重算Delta/coverage、重建I'、mass及mask检查。
- `final_validation_complete.json`：通过测试的源码SHA。
- `protocol_2048.json`：用户修改后的最终参数；`audit_before.json`保留最初256版本任务开始时的状态，二者不要混淆。

源码在项目 `include/hierarchical_representative_si.h`、`src/hierarchical_representative_si.cpp`，synthetic测试在 `tests/hierarchical_representative_si_test.cpp`。CMake测试名 `hierarchical_representative_si`。

完整可复验输入、输出、隔离构建脚本、变更前256工程记录及manifest保存在：

- 本地：`H:/AIProjects/cartgs优化更细致版/hierarchical-representative-si-20261019`
- 服务器：`/home/prmi/zwj/hierarchical-representative-si-20261019`
- 完整包：`hierarchical_si_validation_bundle.tar.gz`

本目录为小型证据归档，不自包含checkpoint/完整输入或runtime。`verify_smoke.py`应在恢复完整包后的独立目录运行；它不会训练或渲染，输出含平台间正常的FP64末位差异。

当前限制：AABB较大，L3 unresolved为28/48，并承载约97.7% Raw SI；mass加权coverage约11.84%。不得把守恒与实现通过解释为代表能力或重建质量已足够。
