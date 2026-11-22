"""独立候选：只移除H2D后的Host等待，保持同Stream依赖与逆矩阵检查。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'camera_graph_preparation.py').read_text(encoding='utf-8')
assert source.count('.cuda()')==2
source=source.replace('GraphCameraPreparer','AsyncGraphCameraPreparer').replace('.cuda()', '.cuda(non_blocking=True)')
source=source.replace('冻结推理相机执行图；保持原BMM、inverse与逐帧错误检查。',
    '异步上传的独立相机候选；不修改冻结Q，不取消逆矩阵错误检查。')
target=HERE/'camera_async_preparation.py';assert not target.exists();target.write_text(source,encoding='utf-8')
edges=(HERE/'camera_graph_edges.py').read_text(encoding='utf-8').replace('from camera_graph_preparation import GraphCameraPreparer',
    'from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer')
edges=edges.replace('CAMERA_GRAPH_EDGES_COMPLETE','ASYNC_CAMERA_EDGES_COMPLETE')
(HERE/'camera_async_edges.py').write_text(edges,encoding='utf-8')
study=(HERE/'camera_graph_study.py').read_text(encoding='utf-8')
begin=study.index('class GraphCameraPreparer(');end=study.index('\n\n@torch.no_grad()',begin)
study=study[:begin]+'from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer\nfrom camera_graph_preparation import GraphCameraPreparer as SynchronousGraphCameraPreparer\n'+study[end:]
study=study.replace('fixed=CameraPreparer(MiniCam);graph=', 'fixed=SynchronousGraphCameraPreparer(MiniCam);graph=')
study=study.replace("['F','camera_graph']", "['Q','async_upload']").replace("name=='F'", "name=='Q'")
study=study.replace('CAMERA_GRAPH_STUDY','ASYNC_CAMERA_STUDY')
(HERE/'camera_async_study.py').write_text(study,encoding='utf-8')
print('ASYNC_CAMERA_CANDIDATE_CREATED')
