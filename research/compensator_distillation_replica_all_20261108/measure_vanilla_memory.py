"""加入CaRtGS原版三次已完成地图；共同RGB帧、各自保存位姿，同Renderer。"""
import measure_lod_memory as memory_test
import scene_runtime as runtime
from scene_runtime import *
import argparse
import subprocess

VANILLA=Path('/home/prmi/zwj/realLOD/saved_experiments/vanilla_all_datasets_20261031_2011/jobs/replica/rgbd')


def configs():
    cases=[]
    for scene_name in ['office2','room0']:
        ours=next(s for s in SCENES if s['scene']==scene_name and s['mode']=='rgbd')
        _,views=full_trajectory_views(Path(ours['checkpoint']),Path(ours['dataset']))
        indices=[19,len(views)//5,2*len(views)//5,3*len(views)//5,4*len(views)//5,len(views)-20]
        rgb_ids=[views[i]['rgb_index'] for i in indices]
        models=sorted((VANILLA/scene_name).glob('*/seed0/*/model/*_shutdown'))
        assert len(models)==3
        for model in models:
            checkpoint=model/'ply'
            repeat=model.parents[3].name
            ply=list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'));assert len(ply)==1
            with ply[0].open('rb') as f:
                count=None
                while True:
                    line=f.readline().decode('ascii').strip()
                    if line.startswith('element vertex'):count=int(line.split()[-1])
                    if line=='end_header':break
            assert count
            item=dict(ours,slug=f'Vanilla__{scene_name}__rgbd__{repeat}',id=f'CaRtGS/{scene_name}/rgbd/{repeat}',
                gaussians=count,checkpoint=str(checkpoint),trajectory=str(model.parent/'CameraTrajectory_TUM.txt'),
                original_ours_id=ours['id'],comparison_rgb_indices=rgb_ids)
            cases.append(item)
    return cases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slug');args=parser.parse_args()
    cases=configs()
    if args.slug:
        case=next(s for s in cases if s['slug']==args.slug)
        runtime.BY_CHECKPOINT[Path(case['checkpoint']).resolve()]=case
        memory_test.SCENES.append(case)
        _,views=full_trajectory_views(Path(case['checkpoint']),Path(case['dataset']))
        lookup={v['rgb_index']:i for i,v in enumerate(views)}
        indices=[lookup[i] for i in case['comparison_rgb_indices']]
        protected=[Path(case['checkpoint'])/'cameras.json',Path(case['trajectory']),
            *Path(case['checkpoint']).glob('point_cloud/iteration_*/point_cloud.ply')]
        before={str(p):sha(p) for p in protected}
        memory_test.worker(case['slug'],'full_map_only',args.output,indices)
        assert before=={str(p):sha(p) for p in protected}
        d=json.loads(args.output.read_text());d.update(case=case,matched_rgb_frames=True,input_sha256=before)
        save(args.output,d);return
    args.output.mkdir(parents=True,exist_ok=False)
    for case in cases:
        p=args.output/(case['slug']+'.json')
        with p.with_suffix('.log').open('x') as f:
            ret=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--slug',case['slug'],
                '--output',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        assert ret.returncode==0,p.with_suffix('.log').read_text()[-1500:]
        print('VANILLA_MEMORY_DONE',case['id'],case['gaussians'],flush=True)
    save(args.output/'COMPLETE.json',dict(passed=True,cases=cases,matched_rgb_frames=True,
        each_map_uses_own_trajectory=True,same_inference_renderer=True,quality_matched=False,
        native_cpp_viewer_vram_measured=False,saved_images=0,models_changed=False))


if __name__=='__main__':main()
