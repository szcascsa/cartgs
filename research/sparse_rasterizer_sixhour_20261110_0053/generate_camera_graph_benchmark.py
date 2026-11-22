"""沿用固定内参验收协议生成独立Q候选对照，不覆盖已冻结脚本。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
target=HERE/'final_camera_graph_benchmark.py';assert not target.exists()
source=(HERE/'final_preparation_benchmark.py').read_text(encoding='utf-8')
source=source.replace('from camera_preparation import CameraPreparer',
    'from camera_preparation import CameraPreparer\nfrom camera_graph_preparation import GraphCameraPreparer')
source=source.replace('FROZEN_PREPARATION.json','FROZEN_CAMERA_GRAPH.json')
source=source.replace('factory=CameraPreparer(MiniCam);b=baseline_B();',
    'factory=CameraPreparer(MiniCam);graph_factory=GraphCameraPreparer(MiniCam);b=baseline_B();')
needle="        for condition_index,condition in enumerate(conditions):"
replacement="""        allocated_before=torch.cuda.memory_allocated();reserved_before=torch.cuda.memory_reserved()
        tick=time.perf_counter_ns();graph_camera=graph_factory.from_source(source,views[0][1]);torch.cuda.synchronize()
        sinks['graph_startup'].add(dict(scene=scene['id'],round=args.round,first_graph_camera_ms=(time.perf_counter_ns()-tick)/1e6,
            allocated_delta=torch.cuda.memory_allocated()-allocated_before,reserved_delta=torch.cuda.memory_reserved()-reserved_before,
            persistent_tensor_bytes=sum(t.numel()*t.element_size() for t in [graph_factory.projection,graph_factory.view,graph_factory.full,graph_factory.inverse,graph_factory.info])))
        for condition_index,condition in enumerate(conditions):"""
assert source.count(needle)==1;source=source.replace(needle,replacement)
source=source.replace("['samples','correctness','startup']","['samples','correctness','startup','graph_startup']")
source=source.replace('camera=source.camera(frame);fixed=factory.from_source(source,frame)',
    'camera=source.camera(frame);fixed=graph_factory.from_source(source,frame)')
source=source.replace("rotate(['A','B','G','F'],", "rotate(['A','B','G','F','Q'],")
source=source.replace("rotate(['original','fixed_intrinsics'],", "rotate(['original','fixed_intrinsics','camera_graph'],")
needle="camera=factory.from_source(source,frame) if name in ['F','fixed_intrinsics'] else source.camera(frame)"
assert source.count(needle)==1
source=source.replace(needle,"camera=graph_factory.from_source(source,frame) if name in ['Q','camera_graph'] else (factory.from_source(source,frame) if name in ['F','fixed_intrinsics'] else source.camera(frame))")
source=source.replace('assert factory.rebuilds==1 and ', 'assert factory.rebuilds==1 and graph_factory.rebuilds==1 and ')
source=source.replace('del source,b,g,factory,references', 'del source,b,g,factory,graph_factory,references')
source=source.replace('FINAL_PREPARATION','FINAL_CAMERA_GRAPH')
target.write_text(source,encoding='utf-8');print('CAMERA_GRAPH_BENCHMARK_CREATED',target)
