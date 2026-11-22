# 暂停现场：2026-10-18

用户明确要求“先停一下等会继续”。本轮已暂停，无本轮后台 benchmark/build 任务。正式生产源码未更新，暂停时对冻结 manifest 的34个生产文件逐一复核，`production_drift=[]`。不要自动继续，等待用户恢复。

## 任务范围

继续 TF Student multi-ratio、evaluation、scalar sync 性能优化；保留已部署 Transport V2、Stage A/B/C、独立 epochs、FP32、四 ratios、B8192、LC6207、E5/20steps。不得继续优化 transport，也不得引入旧 BMM/CUDA Graph。最终报告尚未生成：`CHANGE_DISTILL_TF_MULTI_RATIO_OPTIMIZATION_REPORT.md`。

本机目录 `H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/change_distill_tf_multi_ratio_20261018`。
远端目录 `/home/prmi/zwj/mycartgs/results/change_distill_tf_multi_ratio/20261018`。
全部 raw PT、CSV、JSON、Nsight、build commands、失败日志保存在远端，未覆盖旧审计。`initial.diff`、`initial_status.txt`、baseline source manifest 已保存。所有候选目前只在隔离 source 快照内。

## 已完成及候选结论

- Phase0 新 baseline warm median：TF200.708923ms、Selector195.032267ms、both375.8539075ms。Nsight20452 kernels、19830个<10us；MLP forward38.358ms、Hex forward3.337ms、queryGradKernel9.186ms、Adam0.960ms。详见 PHASE0.md。
- A0 `a0_prepare`：同 batch 共享 outside filtering 和 input gather，训练仍逐 ratio 原 forward/backward。E5完整正确性通过，首轮TF192.556ms、both367.557ms。尚需最终 global/lifecycle（最终组合统一跑）。
- A1 `a1_spatial`：只共享 XY/XZ/YZ sampling，保持六因子乘法次序和原逐 ratio backward。正确性通过，但正反配对TF190.529ms vs A0 190.094ms，无收益，拒绝。其源码/结果保留；最终不应包含 queryMany/CachedQueryFunction。
- C0 `c0_mlp_full`：4B MLP forward + backward，扩展 batch 后改走已有 cuBLAS。LC6207正确性通过，配对TF187.196vs A0 191.367ms、both365.485vs368.831ms。但 global fixture 最终p22失败：max6.03497e-5、gate ratio1.15345。**拒绝**。
- C1 `c1_mlp_safe`：批量 cuBLAS forward，CachedNeuralFunction 仍按原 B/ratio 顺序 backward。LC6207通过；global最终p22 max6.44773e-5仍失败，**拒绝**。
- C2 `c2_mlp_warp`：4B MLP forward保留原基于 B≤10000 的 Warp32选择，MLP backward合并；HexPlane query/backward仍逐ratio。LC6207 TF及both、global/AABB4000/AABB5000完整训练均通过。TF最终最大差1.44392e-5；feature/feature gradient/Teacher targets逐位相同；MLP首步gradient max7.45e-9、plane gradient4.66e-10、first update2.91e-11。**性能待有效测量，不得已宣称保留或部署**。

## 重要源码事实

六平面实际顺序 `XY,XZ,X-ratio,YZ,Y-ratio,Z-ratio`，ratio直接进入sampling；无后置ratio embedding。完整 feature不能共享，也不能将四份 dFeature简单求和后使用一个相同Jacobian。原 first-step autograd顺序实测[3,2,1,0]，C2保持。

## 性能测试暂停位置

另一项目 `/home/prmi/zwj/mycartgs-maturity-validation-20261018` 正在构建/测试/运行SLAM，不能停止它。
本轮C2第一次 native TF 与其GPU tests重叠，数据已重命名为 `c2_mlp_warp/raw/native_tf_contaminated`，**不能用于性能结论**。第二次 retry 在正式测量前因外部GPU进程而退出，见 `c2_native_retry.log`，没有有效 `native_tf/native_both`。

run.py已增加 CMake/CTest/编译/GPU 持续空闲6秒 gate，并监测全程外部PID和编译。仍可能有空闲窗口后的外部新任务，任何监测失败都要保留污染数据并重跑；不能计入最终配对。暂停时本轮没有活跃任务，只有另一项目run.py在运行。

## 恢复后的顺序

1. 确认用户已恢复授权；确认GPU无其他实验、CMake/CTest/build结束。
2. 重跑 C2 `run.py c2_mlp_warp:tf:native_tf:native:10 c2_mlp_warp:both:native_both:native:10`，再 `paired.py c2_mlp_warp a0_prepare` 和 `controls.py a0_prepare c2_mlp_warp`。仅在完整TF/both可重复收益后保留C2。
3. 若C2保留，使用 `evaluation_candidate.py c2_mlp_warp d2_evaluation reuse`，构建/verify/完整训练/native/compare/lifecycle，独立验收evaluation reuse。若拒绝C2，则以A0为base（`d1_evaluation`已生成但未执行）。`d0_evaluation`是基于已拒绝C0的未执行准备稿，不使用。
4. evaluation通过后：`evaluation_candidate.py BASE e0_scalar scalar`，延迟eval312次scalar readback到8次，保留原CPU double fold顺序。训练finite/positive control读回保留。
5. 可额外测 `inside_candidate.py BASE f0_inside`：只有现有nonzero已证明全部inside时，省去identity gather/scatter。单独正确性/性能gate。
6. Adam<1%，不优化；HexPlane backward原kernel约9ms，若最终并非主瓶颈，不大改atomic kernel。
7. 组合为final，完整primitive R1/2/4、B0/1/7/5107/8192/outside，LC/global/AABB，verifyTF/both，正反10warm baseline/final/Selector，controls、Nsight、componentprofile。
8. 正式部署前用manifest防并发修改。`deploy.py`已适配从final/baseline diff自动挑选文件，禁止transport/Selector变更，backup正式lib，cmake完整build、CTest，formal runner配对。已知ultrafast_mapping_cuda旧失败不要修无关模块。
9. `production_check.py`适配paired_production目录；生成中文报告、源码/配置/二进制哈希、精确diff和全部raw归档，再同步所改生产文件到Windows checkout（manifest检查）。不commit。

## 工具及脚本注意事项

- SSH用已有helper `tests/diagnostics/change_distill_performance_20261017/ssh_session.py` +paramiko password hidden input；本轮暂停将关闭SSH会话。不要持久化凭据。
- 服务器必须显式用 `/home/prmi/anaconda3/envs/zwj/bin/python`，默认python为2。
- `build.py stage native|verify|profile|primitive` 复用本轮冻结的正式ABI1 CUDA archives，按源码/headers哈希重编变化CUDA TU，--fmad=false。
- `instrument.py stage verify|profile` 要在对应build前执行。verify捕获全Teacher targets、40step losses（cold+warm）、first feature/gradient、全部TF参数grad/update及full-precision pre/post/audit。
- compare.py现检查verification、verification_final和native_final；原tol不变。
- paired.py已将目录改成 `paired_CANDIDATE_forward|reverse_BRANCH`，避免旧配对目录冲突。每版本每branch各5正+5反warm。
- native runner只保存warm_0 final_parameters，不能假设warm_1..9有PT。native_parity.py已修正本地版本但远端可能仍旧；使用controls.py比较正/反两个warm_0。
- lifecycle.py候选参数可选；复用已存在验证文件，并在assert前保存失败JSON。CPU torch.set_num_threads(1)已加入lifecycle/compare，避免CPU分析大量线程开销。
- primitive.cpp已有shape/forward/outside对比和CUDA-event micro timing；build.py按header自动定义CD_HAS_FORWARD_MANY。C0旧primitive294项通过，C2/最终primitive尚未构建运行。
- final profile需使用最新instrument.py/nsys_analysis.py（本地已加mlp_many_forward与延迟scalar scope，远端尚未上传最新版）。
- `deploy.py`及production_check.py只在本地准备，尚未执行或上传。formal runner来自前轮deployment，动态链接ABI不变。
- 不要执行带 `while pgrep -f "python lifecycle.py ..."` 的等待模板，可能自匹配；此前d0_prepare空稿没有活跃进程，不需要清理生产文件。
