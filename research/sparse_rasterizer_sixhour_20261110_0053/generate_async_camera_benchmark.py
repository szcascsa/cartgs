"""R候选沿用已锁定的24组、14条件和相同配对调度。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'final_camera_graph_benchmark.py').read_text(encoding='utf-8')
source=source.replace('from camera_graph_preparation import GraphCameraPreparer',
    'from camera_graph_preparation import GraphCameraPreparer\nfrom camera_async_preparation import AsyncGraphCameraPreparer')
source=source.replace('FROZEN_CAMERA_GRAPH.json','FROZEN_CAMERA_ASYNC.json')
source=source.replace('graph_factory=GraphCameraPreparer(MiniCam);b=',
    'graph_factory=GraphCameraPreparer(MiniCam);async_factory=AsyncGraphCameraPreparer(MiniCam);b=')
source=source.replace("['samples','correctness','startup','graph_startup']", "['samples','correctness','startup','graph_startup','async_startup']")
needle='        for condition_index,condition in enumerate(conditions):'
replacement="""        allocated_before=torch.cuda.memory_allocated();reserved_before=torch.cuda.memory_reserved()
        tick=time.perf_counter_ns();async_camera=async_factory.from_source(source,views[0][1]);torch.cuda.synchronize()
        sinks['async_startup'].add(dict(scene=scene['id'],round=args.round,first_graph_camera_ms=(time.perf_counter_ns()-tick)/1e6,
            allocated_delta=torch.cuda.memory_allocated()-allocated_before,reserved_delta=torch.cuda.memory_reserved()-reserved_before,
            persistent_tensor_bytes=sum(t.numel()*t.element_size() for t in [async_factory.projection,async_factory.view,async_factory.full,async_factory.inverse,async_factory.info])))
        for condition_index,condition in enumerate(conditions):"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
source=source.replace('camera=source.camera(frame);fixed=graph_factory.from_source(source,frame)',
    'camera=source.camera(frame);fixed=async_factory.from_source(source,frame)')
source=source.replace("rotate(['A','B','G','F','Q'],", "rotate(['A','B','G','F','Q','R'],")
source=source.replace("rotate(['original','fixed_intrinsics','camera_graph'],", "rotate(['original','fixed_intrinsics','camera_graph','async_upload'],")
needle="camera=graph_factory.from_source(source,frame) if name in ['Q','camera_graph'] else (factory.from_source(source,frame) if name in ['F','fixed_intrinsics'] else source.camera(frame))"
replacement="camera=async_factory.from_source(source,frame) if name in ['R','async_upload'] else (graph_factory.from_source(source,frame) if name in ['Q','camera_graph'] else (factory.from_source(source,frame) if name in ['F','fixed_intrinsics'] else source.camera(frame)))"
assert source.count(needle)==1;source=source.replace(needle,replacement)
source=source.replace('and graph_factory.rebuilds==1 and ', 'and graph_factory.rebuilds==1 and async_factory.rebuilds==1 and ')
source=source.replace('del source,b,g,factory,graph_factory,references', 'del source,b,g,factory,graph_factory,async_factory,references')
source=source.replace('FINAL_CAMERA_GRAPH','FINAL_CAMERA_ASYNC')
target=HERE/'final_camera_async_benchmark.py';assert not target.exists();target.write_text(source,encoding='utf-8')
print('ASYNC_CAMERA_BENCHMARK_CREATED')
