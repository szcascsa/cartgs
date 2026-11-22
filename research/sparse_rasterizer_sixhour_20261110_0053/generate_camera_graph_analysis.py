"""复用相同统计口径生成Q的独立验收脚本。"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'analyze_preparation.py').read_text(encoding='utf-8')
source=source.replace('preparation01,preparation02','camera_final01,camera_final02')
source=source.replace("['A','B','G']", "['A','B','G','F']")
source=source.replace("variant=='F'", "variant=='Q'")
source=source.replace("['pass_A','pass_B','pass_G']", "['pass_A','pass_B','pass_G','pass_F']")
for old,new in [('FINAL_PIPELINE_RESULTS','FINAL_CAMERA_GRAPH_PIPELINE_RESULTS'),
    ('PREPARATION_ROUND_RESULTS','CAMERA_GRAPH_ROUND_RESULTS'),('FINAL_PIPELINE_FULL_GATES','FINAL_CAMERA_GRAPH_FULL_GATES'),
    ('FINAL_PREPARATION_RESULTS','FINAL_CAMERA_GRAPH_PREPARATION_RESULTS'),('PREPARATION_CORRECTNESS','CAMERA_GRAPH_CORRECTNESS'),
    ('FINAL_PIPELINE_COUNT_VS_FULL','FINAL_CAMERA_GRAPH_COUNT_VS_FULL'),('PREPARATION_FINAL_VERIFICATION','CAMERA_GRAPH_FINAL_VERIFICATION')]:source=source.replace(old,new)
source=source.replace('F与A/B/G来自','Q与A/B/G/F来自')
source=source.replace("max_speedup_G=group.speedup_G.max()))", "max_speedup_G=group.speedup_G.max(),\n            min_speedup_F=group.speedup_F.min(),median_speedup_F=group.speedup_F.median(),max_speedup_F=group.speedup_F.max()))")
source=source.replace("    out=dict(passed=", "    graph_memory=pd.concat([pd.read_csv(HERE/'reports'/name/'graph_startup.csv') for name in names],ignore_index=True)\n    graph_memory.to_csv(HERE/'CAMERA_GRAPH_MEMORY.csv',index=False)\n    out=dict(passed=")
target=HERE/'analyze_camera_graph.py';assert not target.exists();target.write_text(source,encoding='utf-8')
print('CAMERA_GRAPH_ANALYSIS_CREATED',target)
