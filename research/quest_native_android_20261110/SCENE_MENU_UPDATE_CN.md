# Quest场景菜单与手柄交互更新

本次独立更新只涉及 `research/quest_native_android_20261110`。版本由0.1升级至 `0.2.0-scene-menu`，包名仍为 `ai.mycartgs.quest`，使用原debug证书，支持覆盖安装。原0.1 APK保留且SHA256未变化。

## 使用方式

1. 打开应用，出现前方悬浮的 `MYCARTGS / SCENES` 场景面板，不再显示示例三原色方块。
2. 任意手柄指向场景行，按钮悬停高亮；按食指 Trigger 选择，松开后再点击 `LOAD SCENE`。
3. 加载期间显示 `LOADING SCENE`，后台校验地图、Teacher、Student32、UID和Anchor映射，头部追踪和菜单渲染继续执行。
4. 加载成功进入详情页，显示实际Gaussian/Anchor数量、资产校验结果及推理状态。加载失败显示原因，可返回场景列表。
5. 详情页可点击10% / 20% / 30% / 40% / 50%，或1K / 2K / 3K / 5K / 10K / 15K。当前MLP未运行，按钮只修改请求预算，实际选中点数保持0。
6. 点击 `SCENE LIST` 返回列表；左手柄菜单键返回列表并将面板重新放置在当前头部前方。系统若接管此按键，仍可使用界面按钮返回。

目前实际打包的场景只有 `REPLICA / OFFICE2 / RGB-D`。界面支持64项可信场景目录、每页4项，并具有上一页/下一页；没有伪造未打包的场景。增加其他场景需同时提供对应 `.mcgs`、32/32 Student及地图绑定信息并重新打包，本轮没有重新训练或替换checkpoint。

## 实际改动

- `scene_menu.*`：原生文字网格、面板几何、双手柄射线、悬停、点击和分页；使用当前预测显示时间的输入快照，两眼共享同一次交互结果。
- Oculus Touch兼容绑定改用标准 `input/aim/pose` 与 `input/trigger/value`。原侧握轮转操作已被面板按钮替代。
- 点击采用按下沿和松开重置；失焦、无效姿态、非有限输入、追踪丢失后必须先松开Trigger，不会在恢复后误触。
- `scene_asset.*`：使用打包的 `scenes/catalog.json` 选择场景，继续核验整体及所有section SHA、Teacher fingerprint、Student checkpoint SHA、3534参数Student32、UID与Anchor映射。目录最多64KiB/64项，禁止路径穿越。
- `quest_runtime.*`：启动只读取目录；点击加载后启动一个异步校验任务。最多保留一个场景，重新加载前释放旧场景；Activity退出时等待loader任务结束。
- Vulkan路径：不再生成示例立方体；用一个有界、双面顶点Buffer绘制面板、文字和两手柄射线。容量65,536顶点，即1.5MiB；相同帧的顶点只上传一次，两眼分别绘制，沿用原提交完成同步。
- UI将请求预算与实际选点结果分开：`requestedK`来自界面，`selectedK=0`表示未执行推理，不再把预算伪装成选点结果。

## 验证

| 验证项 | 结果 |
|---|---|
| 原生核心测试 | PASS |
| 双手柄Trigger、悬停及松开去重 | PASS，主机输入模拟 |
| 失焦、追踪丢失/恢复 | PASS，恢复时不产生点击 |
| 面板中心/旋转/背向/平行/NaN射线 | PASS |
| 64场景分页、空目录和有界顶点数 | PASS |
| Android NDK arm64-v8a实际构建 | PASS |
| APK V2签名及与旧版同证书 | PASS |
| APK实际场景33个section与绑定 | PASS |
| 旧APK及原`.mcgs`哈希 | 未变化 |
| Quest 3新版菜单/手柄真机验收 | 未执行，当前ADB列表为空 |

主机从实际UI顶点生成了 `evidence/scene_menu/*_host_preview.png`，已检查文字和按钮布局；这些图片是主机网格预览，不是真机截图。

构建日志、单元测试日志、签名日志及完整哈希在 `evidence/SCENE_MENU_*.log` 和 `evidence/SCENE_MENU_VALIDATION.json`。APK为 `delivery/mycartgs-quest-scene-menu-debug.apk`。

## 仍然存在的功能边界

用户已确认旧版能在Quest 3启动，但本机当前没有ADB连接，因此不能声称新版已经在设备上验证。

当前QNN SDK、API adapter与HTP context仍未完成；Gaussian live draw也未接通。菜单会明确显示 `ASSET ... VERIFIED`、`RENDERING BLOCKED` 和具体QNN原因。选择场景可以加载并核验真实数据，但还不能显示Gaussian实景；没有加入CPU/GPU MLP回退。

源码中未找到 `No connections` 文本，本应用也不需要网络服务器。没有设备日志或系统覆盖层信息时，无法确定该提示的来源。

本轮未修改生产默认、训练路径或原checkpoint，未安装到设备，未commit/push。
