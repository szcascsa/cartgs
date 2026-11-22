from prep_study import *

path=HERE/'reports/FROZEN_JOINT_CANDIDATE.json';assert not path.exists()
binary=Path(dp.deployment_extension().__file__)
for name in ['smoke02','quality_quick01','camera_smoke02','quality_native_quick01','tune01','tune_camera02','projection_compare01','projection_compare02','register01']:
    job=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text());assert job['returncode']==0,name
result=dict(frozen_at=time.time(),selector=2,compensator=2,warps=4,comp_warps=2,
    frustum=1,projection=1,device_schedule=1,native_camera=True,
    binary=str(binary),binary_sha256=sha(binary),
    sources={str(x.relative_to(HERE)):sha(x) for x in (HERE/'source').iterdir() if x.is_file()},
    scope='训练结束后冻结推理，C网络＋CUDA视锥融合＋FP32线性预投影＋GPU动态N/K＋C++当前相机',
    selected_on_development_only=True,development=TUNE,heldout='其余Replica13组',
    before_nonlinearity_only=True,cache_dependencies=['Master offset','Anchor owners','Base/Residual Feature tables','input-layer weights','topology'],
    cache_invalidation='私有冻结快照；地图、网络、Feature或拓扑改变必须新建实例。更改选项清除Graph并重建投影。',
    cache_capacity='N×32＋(levels−1)×anchors×32＋anchors×32 FP32，无视角/Ratio历史缓存',
    rejected=['TF32 frustum: candidate mask changed','per-point projected input: no reproducible pipeline gain','projected register MMA: no pipeline gain'],
    precision='静态预投影FP32；网络FP16乘法FP32累加；当前相机C++ FP64/FP32；物理解码FP32',
    training_modified=False,production_deployed=False)
save(path,result);print('JOINT_CANDIDATE_FROZEN',result['binary_sha256'],flush=True)
