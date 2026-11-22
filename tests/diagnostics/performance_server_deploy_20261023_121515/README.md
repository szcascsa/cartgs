# 服务器性能优化版部署记录

日期：2026-10-23。目标：`prmi@172.19.163.58:/home/prmi/zwj/mycartgs`。

## 部署结果

- 清单内 306 个本地源码、配置、入口及必要测试文件与服务器 SHA-256 完全一致；实际新增或覆盖 48 个文件。
- 七项 CUDA 优化均默认开启，服务器 OpenCV FileStorage 解析确认值全部为 `1`；显式设为 `0` 可逐项回退 Legacy。
- 使用 CUDA 11.8、sm_86、g++9、LibTorch 2.3.1 ABI1，在主目录独立构建 `build-performance-20261023-121515`。
- 17 个目标构建成功，包括六个训练入口、两个离线入口、七个 CUDA 模块测试、R3 生命周期测试和原生重放工具。
- 11 个运行产物保存 SHA-256；`ldd` 无缺失依赖，不引用隔离实验目录中的动态库。TUM RGB-D 入口的缺少必要参数检查正常返回。
- Adam、Decode/VJP、Input27/VJP、Anchor 统计、Selector Sampling/Stable Compaction、Renderer workspace、SSIM Backward 测试全部通过。
- R3/Variable Anchor 生命周期测试通过，覆盖 clone/split、prune、growing、Adam、几何迁移与 checkpoint；15 项启动器测试通过。

此次为已授权版本同步及部署验证，没有重新运行完整在线训练，也不替代此前报告中的长期一致性与极端数值边界结论。未修改历史实验配置、模型及结果目录；没有 stage、commit 或 push。

## 运行入口与备份

```bash
cd /home/prmi/zwj/mycartgs
bash run_interactive.sh
```

旧源码及二进制备份：`/home/prmi/zwj/performance-deploy-20261023-121515/backup`。备份中的 `source/` 对应被覆盖且原先存在的文件，`artifacts/` 保存原 `bin/`、`lib/` 文件；新增文件与旧文件的区别记录在 `changes.json`，没有执行删除旧文件或清理结果目录。

## 证据

- `manifest.json`：部署文件及本地 SHA-256。
- `verification.json`：部署后源码、开关、11 个产物 SHA-256 及构建状态。
- `deployment_evidence.zip`：编译、运行、数值验证日志与 Parity CSV；包含首次 CMake 配置缺少 jsoncpp 搜索路径的失败记录，以及补齐既有依赖路径后的成功记录。
- `deploy.py`：本次服务器部署脚本，仅作用于清单文件、构建目录及验证目录。
