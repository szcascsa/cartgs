"""最新Graph对照的同输入配对微测，只用于5个调优场景。"""
from study_common import *
from footprint_runtime import Renderer,MODES,extension


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--validation',required=True)
    parser.add_argument('--modes',default='0,1,2,3,4,5,6,7');parser.add_argument('--repeats',type=int,default=4)
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    validity=json.loads((HERE/'reports'/args.validation/'SUMMARY.json').read_text());assert validity['passed']
    assert all(MODES[k] in validity['valid_modes'] for k in map(int,args.modes.split(','))),'Unvalidated candidate requested'
    rows=[];events=[];checks=0
    for scene_index,scene in enumerate(SCENES):
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug']);previous=previous_renderer()
        engines={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
        engines={'previous_G':previous,**engines};views=views_for(source,'tune')[:2]+views_for(source,'tune')[-2:]
        for engine in engines.values():engine.precompile(source.runtime.total,source.camera(views[0][1]),source.scene.background)
        for condition_index,condition in enumerate(['local0.01','local0.05','local0.1','local0.7','full']):
            load=external_load();assert not load['active'],load
            packets=[]
            for motion,frame in views:
                camera=source.camera(frame);packet,ids,_=prepare(source,camera,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for name,engine in engines.items():
                    image,radii=engine.render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']),(scene['id'],frame,condition,name)
                    checks+=1
                packets.append((motion,frame,camera,packet,ids))
            for repeat in range(args.repeats):
                for index in range(16):
                    motion,frame,camera,packet,ids=packets[index%len(packets)]
                    for name in rotate(list(engines),index+repeat+scene_index+condition_index):
                        torch.cuda.synchronize();tick=time.perf_counter_ns();image=engines[name].render(camera,packet,source.scene.background)[0]
                        torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                        rows.append(dict(scene=scene['id'],condition=condition,variant=name,repeat=repeat,iteration=index,
                            frame=frame,motion=motion,K=len(ids),N=source.runtime.total,wall_ms=elapsed))
            for name,engine in engines.items():
                samples=[]
                for index in range(12):
                    _,frame,camera,packet,ids=packets[index%len(packets)]
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();image=engine.render(camera,packet,source.scene.background)[0];end.record();end.synchronize();samples.append(start.elapsed_time(end))
                events.append(dict(scene=scene['id'],condition=condition,variant=name,K=len(ids),event_ms=st.median(samples),
                    workspace_bytes=engine.native.workspace_bytes()))
            load=external_load();assert not load['active'],load
            write_csv(out/'samples.csv',rows);write_csv(out/'events.csv',events)
            print('FOOTPRINT_SCREEN',scene['id'],condition,len(rows),flush=True)
        del source,engines,previous,packets,packet,ref,image;torch.cuda.empty_cache()
    save_json(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=checks,modes=args.modes,
        scope='5组调优集；旧最新Graph为previous_G，新control独立核对等价，原A为数值oracle',
        binary_sha256=hashlib.sha256(Path(extension().__file__).read_bytes()).hexdigest()))
    print('FOOTPRINT_SCREEN_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
