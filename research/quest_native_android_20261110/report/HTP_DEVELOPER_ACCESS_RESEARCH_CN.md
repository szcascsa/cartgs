# Quest 3 开发者身份与 HTP 访问：公开资料核查

核查日期：2026-11-10。范围为 Meta/Qualcomm 公开文档、相关上游 issue 与当前设备声明的 Android 权限；没有向外部提交消息，没有修改 APK、系统或训练。

## 结论

公开资料中未找到“注册 Meta 开发者后，自助申请或勾选即可启用 Quest 3 HTP/NPU”的权限名称、entitlement 或操作流程。开发者身份有助于使用侧载、调试与开发者支持渠道，不应等同于获得 DSP 设备访问权限。未找到公开入口也不能证明所有合作伙伴或邀请计划都不存在。

本机目前确认的是：QNN 2.40 路径的普通 APK 遇到 DSP 节点访问拒绝，FastRPC HAL 会话也失败。不能进一步推断所有 SDK/固件组合都不可用，或者 Meta 必然有可申请的白名单。

## 来源与证据范围

| 来源 | 可支持的结论 | 不能据此推断的内容 |
|---|---|---|
| Meta Device Setup / Test your app | 开发者模式用于安装、运行、调试商店外应用 | 注册开发者会解除 SELinux 或 HTP 限制 |
| Meta Entitlements API | 检查用户是否有资格使用该应用 | Entitlement 等于 NPU 硬件授权 |
| Meta 社区 Quest 3 NPU 问题 | 社区存在直接访问 NPU 的诉求；读到的页面没有可执行的授权办法 | 官方承诺普通 APK 的 HTP 支持或正式禁止所有访问 |
| Qualcomm FastRPC 技术说明 | 通道依赖 OEM 系统镜像提供的驱动、服务、固件及 vendor 库 | 仅打包 SDK 即可在任何设备获得通道 |
| ExecuTorch #20168 | 用户报告 Quest 3 / SXR2230P / V69 / QNN 2.37 跑过 HTP 路径，但模型结果异常 | 已证明普通 APK、当前固件、无特权环境可用；该版本一定修复我们的权限错误 |
| LiteRT #9754 / PR #9905 | 有其他 Android XR 平台需要 Signed PD，支持已进入 LiteRT | 对 Quest 3 V69 必然适用；Signed PD 能改变 Android 应用权限 |
| Qualcomm ai-hub-apps #157 | 其他手机出现普通 APK 与 shell HTP 可访问性不同；维护者表示调查 | 该手机的诊断可以直接替代 Quest 结论或已有通用修复 |

设备只读核查列出 1,569 个 Android 权限声明，未找到按 HTP/NPU/DSP/QNN/HEXAGON/NEURAL 名称段匹配的条目。证据：`evidence/HTP_DECLARED_PERMISSION_AUDIT.json`。该核查只覆盖权限声明，不覆盖私有服务、合作伙伴计划或未知命名的权限。

## 可执行的后续路径

1. **推荐：使用已注册开发者账号向 Meta Developer Support 确认支持边界。** 附本项目最小复现和 Signed/Unsigned APK 日志，询问当前消费版 Quest 3 是否支持第三方原生 QNN HTP；若支持，要求提供兼容固件、QAIRT 版本、必要 Manifest/API、PD 类型及签名镜像取得方式；若受限，询问是否存在合作伙伴/研究测试申请入口。公开支持页面存在，但不能承诺一定会授予访问。
2. **技术对照：隔离测试 QNN 2.37 与当前 2.40。** 首先只验证普通 APK 中 deviceCreate 和极小固定图，不接完整模型；同时核对 FastRPC HAL 是否存在并可调用。这用于区分 runtime/固件兼容性与平台限制，尚未执行，不能当成已发现的修复。
3. **有条件调查：系统 NNAPI 是否真正暴露 HTP 设备。** 只有枚举到可识别的硬件设备，并强制指定设备、关闭 CPU fallback、证实算子驻留后才有意义。搜索到的 Quest NNAPI benchmark 标注设备为 ARM，不能作为 NPU 成功证据。该路径也不等于已满足当前指定的直接 QNN 部署要求，本轮没有切换后端。

Signed PD 已在我们的普通 APK 上测得相同通道错误，不作为新修复重复推荐。Unity Inference Engine 文档没有提供这个原生 QNN 访问问题的解决方案。网上部分开发板教程依赖 root/SELinux permissive，与当前消费版设备和任务约束不符，未执行。

## 官方支持入口与原始链接

- Meta 开发者支持：<https://developers.meta.com/vr/support/>。
- Meta 如何提报问题：<https://developers.meta.com/vr/blog/how-to-get-help-file-bugs-track-fixes-building-with-meta-horizon/>。
- Meta 设备开发设置：<https://developers.meta.com/vr/documentation/native/android/mobile-device-setup/>。
- Meta Entitlements：<https://developers.meta.com/vr/reference/platform-android/v72/classcom_meta_horizon_platform_ovr_requests_entitlements/>。
- Quest 3 NPU 社区问题：<https://communityforums.atmeta.com/discussions/dev-quest/direct-access-to-quest-3s-neural-processing-unit-qualcomm-hexagon-processor-npu/1311021>。
- Qualcomm FastRPC 说明：<https://www.qualcomm.com/developer/blog/2026/10/how-ai-models-reach-hexagon-npu-qairt-fastrpc-hexagon-explained>。
- Quest QNN 2.37 运行报告：<https://github.com/pytorch/executorch/issues/20168>。
- LiteRT Signed PD：<https://github.com/google-ai-edge/LiteRT/issues/9754>、<https://github.com/google-ai-edge/LiteRT/pull/9905>。
- 其他设备的应用权限差异：<https://github.com/qualcomm/ai-hub-apps/issues/157>。

建议向官方提交的核心技术问题为：普通 `untrusted_app` 上，`libcdsprpc` 已加载且 Stub 已连接，Signed/Unsigned 都在 `open_hal_session` 阶段失败并返回 `QnnDevice_create=14001`；当前固件 `52641960034500520` 应使用哪条受支持的第三方调用路径？
