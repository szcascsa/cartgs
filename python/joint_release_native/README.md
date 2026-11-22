# Mode182 冻结原生实现

此目录提供默认 `JointReleaseRenderer(mode=182)` 的独立 CUDA/C++ 构建来源。`_extension.py` 编译完整的 11 个 translation unit，不依赖研究目录、预编译 `.o` 或共享环境改动。`_network.py` 保留冻结网络快照与管线初始化，仅将原研究加载器替换为包内构建器。

67 份原生源码来自 2026-11-15 验收的 `source_joint_release`，逐文件 SHA256 记录于 `SOURCE_PROVENANCE.json`。`.gitattributes` 禁止该目录的换行转换，保证跨平台 checkout 后身份检查仍然有效。来源二进制哈希只用于追溯；本包不随代码分发该二进制，也不将重新编译的结果宣称为同一二进制。

为保持已验收源码和 include 关系，保留快照中包含的其他候选头文件及参考头文件；公开 Python API 只允许 Mode182。Alpha3、其他阈值和实验模式不会被默认选中。

参考 Rasterizer 的版权声明保留在原文件中，相关许可证见仓库根目录 `LICENSE.md` 和 `LICENSE`。本目录没有重新授予上游代码许可。

使用方法、缓存依赖、单 Stream 约束及精度边界见 `../DEPLOYMENT_RUNTIME.md`。
