# 训练完成后的独立部署推理

`deployment_runtime.py` 是独立的冻结部署入口。正常训练、训练前向、训练评价和现有 `pytorch/cuda/cuda_graph` 入口继续使用原实现；这些入口不会自动导入部署模块。没有训练优化器、反向传播或在线拓扑更新。

部署后端保留FP32、当前网络权重、严格K和固定Gumbel噪声，但MLP的FMA累加、视角编码和Quaternion范数使用专用CUDA实现，允许与cuBLAS产生细小误差。质量验证报告不能称为逐bit等价。

```python
import torch
from deployment_runtime import DeploymentRenderer

torch.backends.cuda.matmul.allow_tf32 = False
runtime = DeploymentRenderer.from_checkpoint('/path/to/finished/23560_shutdown/ply')

# camera沿用项目MiniCam：相机矩阵为CUDA FP32，图像尺寸和FoV与相机标定一致。
with torch.inference_mode():
    result = runtime.render(camera, budget=2000)
    rgb = result['render']       # CUDA FP32 [3,H,W]
    selected_ids = result['ids']
    stats = result['stats']

    # 也可以按当前名义可见Anchor成员数N计算K=floor(ratio*N)。
    result = runtime.render(camera, ratio_budget=0.1)
```

`budget` 和 `ratio_budget` 二选一。预算不足时K=min(请求预算,可选点数)，缺口写入 `budget_shortfall`。radii>0和真实Alpha贡献点数仍可能小于K。旧11D无视角模型需要显式传 `view_conditioned=False`。也支持 `compensation=False` 以检查选择结果，以及 `full()` 获取原始未补偿Full输入。

输入checkpoint需有 `selector.pt`、`selector_config.yaml` 和唯一的 `point_cloud/iteration_*/point_cloud.ply`。当前支持11D／19D Selector、Residual48、64/64主干、XYZ／Scale／Rotation／Other四Head、Softplus或物理加法Scale，兼容joint schema1/2/3/4。其他网络形状会显式拒绝。没有新增Gaussian可训练参数。

加载时复制并冻结独立模型快照，转置权重以改善读取。相机、视锥、ratio编码、Residual查询、Selector和Compensator每次重新计算；没有上帧Mask或ratio→Anchor48结果缓存。更换checkpoint或地图时构造新实例；同一实例由调用者串行调用，多请求应在外层加锁。部署接口拒绝启用梯度的调用。

计算划分：Selector每Gaussian一个Warp；低预算Compensator保留每Gaussian一个Block、四Warp并行Head。高预算使用每Block四Gaussian的层权重共享Tile，每Gaussian仍按原顺序计算主干和四个独立Head。两者同时完成Residual输入读取、ratio和view投影、属性解码与发布，不改变网络结构。

当前v11使用会收缩边界桶的协作式精确Radix，之后按原候选行序稳定发布K个ID。GPU连续准备路径扩展到冻结地图容量不超过65536；更大地图保留C++计数路径。N、K、相机和ratio仍逐请求更新。该容量门限仅在RTX3090和现有checkpoint验证，不代表所有设备的最优门限。stats中的`device_counts`、`shrinking_radix`和`compensation_tile`分别标记本次路径。

不依赖图形界面的独立入口：

```bash
python python/deployment_infer.py /path/to/finished/ply \
  --dataset /path/to/dataset --ratio 0.1 --output /path/to/new_metrics.csv
```

该入口只运行冻结推理，输出逐帧N、K、ratio及墙钟，不训练、不保存渲染图片，也不覆盖已有CSV。原有训练脚本和Viewer默认后端没有切换。

部署版使用C++调度：完整准备在 `deployment_prepare.cpp` 的 `DeploymentPipeline::prepare_native` 中执行，Python每帧只调用一次管线。新路径在候选压缩后由GPU计算本请求的N、合资格数量、目标预算、实际K、ratio编码及Residual权重；Selector、精确选点和Compensator直接读取设备端控制记录。准备末尾将96字节控制记录统一读回一次，再构造精确K的Tensor视图对接现有Rasterizer。容量超过65536时继续沿用C++计数读回路径，没有Python回退。

两项优化可独立关闭：`DeploymentRenderer(..., shrinking_radix=False, compensation_tile=0)`；`from_checkpoint`接受相同参数。`compensation_tile`支持0、4、8，默认4；低于阈值仍自动使用原单点Block。关闭Radix时，超过16384点的地图恢复原Host计数/CUB路径。协作式Radix需要设备支持cooperative launch，尚未验证其他端侧GPU。

C++持有有界GPU工作区与Pinned Host缓冲，GPU等待期间释放GIL。属性和ID按用户预算及整图规模提供的上界分配，Kernel只对有效N/K执行网络和发布运算；容量尾部直接退出。返回结果有独立存储，后续请求不会覆盖此前输出。相机每帧由应用提供，不复用前一帧视锥、选点或ratio特征结果。Rasterizer内部Tile展开计数仍保留原实现的读回，本阶段不宣称整个渲染链无同步。

部署版只有C++调度入口，不提供 `scheduler` 参数或CLI选项。历史版本独立保存在实验参考快照中。正常训练和常规推理入口继续保持原实现。计时应使用外层墙钟；C++返回的 `prepare_ms` 只覆盖native核心，不包含Python入参检查和最终返回封包。

新路径中Selector、设备端选点和Compensator共享本请求的finite状态。同一CUDA Stream保证依赖顺序；Selector异常时选点和补偿在读取数据前退出。Host在准备末尾统一读取并优先报告Selector错误，异常不会进入Rasterizer；关闭补偿时同样保留该检查。大地图保持原C++错误检查方式。

运行环境需要项目现有PyTorch CUDA、Rasterizer、模型读取依赖，以及编译CUDA扩展所需的NVCC／C++编译器／Ninja。扩展第一次加载时独立编译；可用 `TORCH_EXTENSIONS_DIR` 指定部署构建目录，使用目标设备的CUDA架构编译。不要直接把RTX3090编译出的二进制当作其他端侧GPU的验证结果。当前速度证据来自RTX3090，实际端侧设备需要使用同一入口复测。
