# 结果整理与Master固定消融准备报告



baseline已完整迁移32组双网络真实视角＋LI=0.1结果，包含736项全轨迹评价。

vanilla已完整迁移101次成功运行及1次历史跳过，保留Replica、TUM、VECtor和3DLite。

迁移使用同盘rename；旧saved_experiments路径保留符号链接。2928文件SHA256、大小和inode复验通过。



消融包含Replica16组、TUM10组、VECtor6组，637377步Fine、736项评价。

每组绑定baseline使用的最佳原Full checkpoint、相机、seed0和步数，不从Fine后的模型继续。

唯一算法差异：Full／全局LoD／局部LoD三路全部detach Master；只更新Selector、Compensator和Anchor Feature。

Full渲染、原Loss及反向仍执行，Full继续提供LI教师；参数更新仍是三路之后一次共享Adam。

训练ratio=[0.01,0.05,0.1,0.3,0.5,0.7]；未见ratio=[0.03,0.08,0.2,0.4,0.6]只评价。

保留原候选池、预算、LI=0.1、健康停止、OOM重试上限及双目左图校正；不保存渲染图片。



仅修改独立source内3个C++文件，头文件110项逐位一致，未改生产或baseline源码。

独立增量编译3个对象、Mapper动态库和Fine入口成功；其余对象只读复用。

ldd/readelf确认入口加载ablation/runtime/lib/libgaussian_mapper.so，依赖全部找到。

16项CPU测试通过，涵盖单一配置差异、32组Full起点与步数匹配、重试语义和启动锁。

所有6类Master参数设置requires_grad=false；训练结束必须逐位不变、无梯度、无Adam状态。

运行时还必须验证网络更新及Anchor Feature Adam状态存在。



**当前没有启动训练、GPU单元测试或场景smoke。** runs/evaluation/validation均为空；HOLD存在，启动授权不存在。

待用户明确说跑后，监督器先执行GPU验证，再按各数据集/相机入口进行6步smoke和初态逐位校验，通过才正式训练。

尚未执行的GPU和场景验证不构成已通过项；失败时队列停止并保留现场。



启动入口（本次未执行）：`/home/prmi/anaconda3/envs/zwj/bin/python /home/prmi/zwj/realLOD/ablation/start_queue.py --start`。

主队列：`LOD_PIPELINE.json`；协议：`reports/ABLATION_PROTOCOL.md`；验收：`reports/PREPARED.json`。
