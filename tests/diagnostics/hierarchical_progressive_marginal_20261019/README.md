# Progressive / Marginal Tail 离线机制实验

本目录是当前冻结 fr1_room 的完整实验副本。正式 C++ Hierarchical SI、配置、Teacher、Selector 和 renderer 均未修改；新增算法只在 `progressive_ranker.py` 的离线 reference 中运行。

最终判定以 `mechanism_result.json` 和项目根 `HIERARCHICAL_PROGRESSIVE_MARGINAL_AUDIT.md` 为准：F2/F3 达到预注册支持条件，F1 没有达到“明显重复度下降”门槛，F3 没有达到 Strong improvement。建议 ADOPT BOTH 仅表示后续研究候选，没有部署或接入在线路径。

- `mechanism_preregistration.json`：运行前冻结的参数、tie-break、实验 gate。
- `new_arm_rankings.npz`：三条完整排序、ordinal score 和 emit-time priority。
- `ranking_emit_trace.csv`：所有 emit 的真实顺序、身份、depth 与 priority。
- `frozen_tree_membership.npz`、`frozen_tree_nodes.json`：原节点候选、provisional reps 与 children，结构完全冻结。
- `synthetic_tests.json`、`independent_replay_checks.json`：synthetic 与150990次候选/gain/frontier复核。
- `progressive_marginal_quality_summary.csv`、`progressive_marginal_per_view.csv`：六arm四ratio的真实render质量。
- `paired_statistics.csv`、`paired_per_view_deltas.csv`：配对逐view分析。
- `download_verified.json`、`local_independent_validation.json`：下载SHA和Windows独立数值/判定复核。
- `artifact_manifest.json`：完整服务器证据包中116个文件的SHA；本README和本地验证记录为下载后追加。

主实验数据总N=52635、candidate=50330，69 Bank / 161 fixed non-Bank archived evaluation views。完整输入/TF/相机/GT来源见SHA清单；大型原始camera/GT archive继续复用原服务器路径，没有复制进包。

服务器完整目录：`/home/prmi/zwj/hierarchical-progressive-marginal-20261019`。

本地完整目录：`H:/AIProjects/cartgs优化更细致版/hierarchical-progressive-marginal-20261019`。

完整包：`hierarchical_progressive_marginal_evidence.tar.gz`。

仅做排序检查可用NumPy执行 `test_progressive.py` 或 `replay_checks.py`；复现真实render需要脚本所记录的旧LibTorch/CUDA/指标环境、冻结runtime库与外部数据。`prepare.py` 和baseline脚本含原服务器路径，不能在缺失数据时静默换环境。`local_verify.py` 应在与mycartgs同级的完整实验目录运行。

没有commit、没有push、没有性能优化或参数搜索。
