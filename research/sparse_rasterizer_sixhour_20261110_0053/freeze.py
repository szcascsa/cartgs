"""在保留场景性能开放前锁定配置、源代码与实际二进制身份。"""
from pathlib import Path
import argparse,hashlib,json,time

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,default=1);args=parser.parse_args()
    path=HERE/'reports/FROZEN_CANDIDATE.json';assert not path.exists()
    names=['edges01','sanitizer_mem01','motion01','formal_smoke01'] if args.revision==1 else ['edges04','sanitizer_mem04','sanitizer_init04','motion02','formal_smoke02','coverage01']
    checks={name:json.loads((HERE/'reports'/name/'SUMMARY.json').read_text()) for name in names}
    assert all(row['passed'] for row in checks.values())
    assert all(row['final_graph_stats'][1:3]==[0,0] for row in checks['motion01' if args.revision==1 else 'motion02']['startup'])
    sources={str(p.relative_to(HERE)):sha(p) for p in (HERE/'source').rglob('*') if p.is_file()}
    sources.update({name:sha(HERE/name) for name in ['runtime.py','graph_runtime.py','harness.py','final_benchmark.py','diagnostics_check.py']})
    binaries={name:dict(path=str(HERE/path),sha256=sha(HERE/path)) for name,path in [
        ('regular','build/mycartgs_sparse_sixhour_v1.so'),('graph','build_graph/mycartgs_sparse_graph_study_v1.so')]}
    value=dict(frozen=True,performance_tuning_complete=True,time=time.time(),revision=args.revision,
        regular=dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=args.revision>=2),
        graph=dict(variant=5,capacity=256,plan_granularity=512,max_plans=512,graph_input_update=False,ordered_coverage=args.revision>=2,device_clear=args.revision>=2),
        precompile_policy='模型加载按地图容量和分辨率预编译全部512粒度计划；每帧重读所有Gaussian/相机/输出指针',
        max_workspace_bytes=1<<30,max_bitset_bytes=256<<20,source_hashes=sources,binaries=binaries,
        acceptance='Full每场景两轮中位不慢于A；推荐还须不慢于B。RGB/radii bitwise。保留场景不调参。',
        checkpoint_and_selection_unchanged=True,training_unchanged=True,production_defaults_unchanged=True,
        tuning_scenes_only=True,heldout_opened=args.revision>=2,
        baseline_B_historical_numeric_failure=args.revision>=2,
        revision_reason='按原二进制SASS固定Tile coverage FMA舍入顺序；不按场景/Ratio调参；B继续只读作为历史速度对照' if args.revision>=2 else '')
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False));print('CANDIDATE_FROZEN',value['time'],binaries,flush=True)


if __name__=='__main__':main()
