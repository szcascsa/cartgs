"""R候选按相同统计门限独立分析，不替换已有Q证据。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'analyze_camera_graph.py').read_text(encoding='utf-8')
source=source.replace('camera_final01,camera_final02','async_final03,async_final02')
source=source.replace("['A','B','G','F']", "['A','B','G','F','Q']")
source=source.replace("variant=='Q'", "variant=='R'")
source=source.replace("['pass_A','pass_B','pass_G','pass_F']", "['pass_A','pass_B','pass_G','pass_F','pass_Q']")
source=source.replace("max_speedup_F=group.speedup_F.max()))", "max_speedup_F=group.speedup_F.max(),\n            min_speedup_Q=group.speedup_Q.min(),median_speedup_Q=group.speedup_Q.median(),max_speedup_Q=group.speedup_Q.max()))")
source=source.replace('FINAL_CAMERA_GRAPH_PIPELINE_RESULTS','FINAL_ASYNC_PIPELINE_RESULTS')
source=source.replace('CAMERA_GRAPH_ROUND_RESULTS','CAMERA_ASYNC_ROUND_RESULTS')
source=source.replace('FINAL_CAMERA_GRAPH_FULL_GATES','FINAL_ASYNC_FULL_GATES')
source=source.replace('FINAL_CAMERA_GRAPH_PREPARATION_RESULTS','FINAL_ASYNC_PREPARATION_RESULTS')
source=source.replace('CAMERA_GRAPH_CORRECTNESS','CAMERA_ASYNC_CORRECTNESS')
source=source.replace('FINAL_CAMERA_GRAPH_COUNT_VS_FULL','FINAL_ASYNC_COUNT_VS_FULL')
source=source.replace('CAMERA_GRAPH_MEMORY','CAMERA_ASYNC_MEMORY')
source=source.replace("'graph_startup.csv'", "'async_startup.csv'")
source=source.replace('CAMERA_GRAPH_FINAL_VERIFICATION','CAMERA_ASYNC_FINAL_VERIFICATION')
source=source.replace('Q与A/B/G/F来自','R与A/B/G/F/Q来自')
target=HERE/'analyze_camera_async.py';assert not target.exists();target.write_text(source,encoding='utf-8')
trajectory=(HERE/'camera_trajectory_parity.py').read_text(encoding='utf-8')
trajectory=trajectory.replace('from camera_graph_preparation import GraphCameraPreparer',
    'from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer')
trajectory=trajectory.replace('CAMERA_TRAJECTORY','ASYNC_CAMERA_TRAJECTORY')
(HERE/'camera_async_trajectory.py').write_text(trajectory,encoding='utf-8')
profile=(HERE/'profile_final_camera.py').read_text(encoding='utf-8')
profile=profile.replace('from camera_graph_preparation import GraphCameraPreparer',
    'from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer')
profile=profile.replace('FROZEN_CAMERA_GRAPH.json','FROZEN_CAMERA_ASYNC.json').replace("'Q/pipeline/'", "'R/pipeline/'").replace('Q_PROFILE','R_PROFILE')
(HERE/'profile_async_camera.py').write_text(profile,encoding='utf-8')
print('ASYNC_CAMERA_ANALYSIS_CREATED')
