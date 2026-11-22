"""只读加载已验收.so的研究入口；不触发JIT，不修改生产或训练。"""
from pathlib import Path
import hashlib,importlib.util,json
import graph_runtime,runtime

HERE=Path(__file__).resolve().parent


def load_verified(name,relative_path,expected):
    path=HERE/relative_path
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise RuntimeError(f'冻结二进制哈希不符：{path}')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def create_camera_preparer(camera_class,*,variant='Q'):
    """显式选择已通过正式验收的相机准备路径，加载时核对源代码。"""
    names={'F':('FROZEN_PREPARATION.json','PREPARATION_FINAL_VERIFICATION.json'),
        'Q':('FROZEN_CAMERA_GRAPH.json','CAMERA_GRAPH_FINAL_VERIFICATION.json'),
        'R':('FROZEN_CAMERA_ASYNC.json','CAMERA_ASYNC_FINAL_VERIFICATION.json')}
    if variant not in names:raise ValueError('Camera variant must be F, Q or R')
    frozen,result=names[variant]
    if not (HERE/result).is_file() or not json.loads((HERE/result).read_text())['passed']:
        raise RuntimeError(f'相机路径{variant}尚未通过正式验收')
    config=json.loads((HERE/'reports'/frozen).read_text())
    for name,expected in config['sources'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'相机准备源码哈希不符：{name}')
    if variant=='R':
        from camera_async_preparation import AsyncGraphCameraPreparer
        return AsyncGraphCameraPreparer(camera_class)
    if variant=='Q':
        from camera_graph_preparation import GraphCameraPreparer
        return GraphCameraPreparer(camera_class)
    from camera_preparation import CameraPreparer
    return CameraPreparer(camera_class)


def create_renderer(camera,background,maximum_points,*,use_graph=True):
    """一个device/Stream对应一个实例；每帧传入当前packet和camera。"""
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text())
    if not config.get('frozen'):raise RuntimeError('未找到已冻结候选')
    if use_graph:
        module=load_verified('mycartgs_sparse_graph_study_v1','build_graph/mycartgs_sparse_graph_study_v1.so',config['binaries']['graph']['sha256'])
        graph_runtime.extension=lambda:module
        engine=graph_runtime.GraphRenderer(**config['graph']);engine.precompile(maximum_points,camera,background)
    else:
        module=load_verified('mycartgs_sparse_sixhour_v1','build/mycartgs_sparse_sixhour_v1.so',config['binaries']['regular']['sha256'])
        runtime.extension=lambda:module;engine=runtime.SparseRenderer(**config['regular'])
    return engine
