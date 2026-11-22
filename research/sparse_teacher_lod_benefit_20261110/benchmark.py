"""只读使用已验收G/R与原Teacher；补充直接Master Full的严格对照。"""
from pathlib import Path
import sys
REFERENCE=Path('/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053')
sys.path.insert(0,str(REFERENCE))
from harness import *
from frozen_renderer import create_renderer,create_camera_preparer
import threading

OUT=Path(__file__).resolve().parent
SLUGS=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd',
    'TUM_RGBD__freiburg1_xyz__rgbd']
CONDITIONS=['full_direct','full_copy','local0.01','local0.05','local0.1','local0.3','local0.7']


def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


class Monitor:
    def __init__(self):
        self.rows=[];self.stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        while not self.stop.is_set():
            self.rows.append(external_load())
            self.stop.wait(.1)
    def check(self):
        assert not any(r['active'] for r in self.rows),('External GPU activity',self.rows[-3:])
    def close(self):self.stop.set();self.thread.join();self.check()


def rotate(items,i):
    j=i%len(items);out=items[j:]+items[:j]
    return out if (i//len(items))%2==0 else out[::-1]


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reports=OUT/'reports';reports.mkdir(exist_ok=False)
    protected=read_json(REFERENCE/'reports/PROTECTED_BEFORE.json')
    # 以前保护清单用于标识范围，当前重新记录before，不把历史后续变更归于本轮。
    protected={name:sha_file(name) for name in protected}
    cfg=read_json(REFERENCE/'reports/FROZEN_CANDIDATE.json')
    for p in [REFERENCE/'frozen_renderer.py',REFERENCE/'reports/FROZEN_CANDIDATE.json',
            REFERENCE/'reports/FROZEN_CAMERA_ASYNC.json',
            *[Path(v['path']) for v in cfg['binaries'].values()]]:
        protected[str(p)]=sha_file(p)
    write_json(reports/'PROTECTED_BEFORE.json',protected)
    ids=[s['slug'] for s in SCENES]
    for slug in SLUGS:assert slug in ids,(slug,ids)
    assert not external_load()['active'],'GPU not idle'
    monitor=Monitor();samples=[];checks=[];identity=[]
    write_json(reports/'PROTOCOL.json',dict(slugs=SLUGS,conditions=CONDITIONS,rounds=2,repeats=6,
        views_per_scene=18,scope=['raster','pipeline'],renderer='frozen G',camera='frozen R',
        compensator='original64/64 and four two-layer Heads; no student',
        full_direct='fixed Master packet prepared once before timing, current camera every frame',
        full_copy='legacy benchmark runtime.full clone/publish every request',
        topology_weights_and_algorithms_unchanged=True,same_camera_and_background=True,
        quality_equivalence_between_ratios_claimed=False))
    for slug in SLUGS:
        scene,source=load_scene(slug)
        assert source.runtime.student is None
        model=source.runtime._base.model
        assert model.spatial_compensation.encode.out_features==64
        assert model.spatial_compensation.body.out_features==64
        comp_params=sum(p.numel() for p in model.spatial_compensation.parameters())
        assert comp_params==25742,(slug,comp_params)
        identity.append(dict(scene=scene['id'],slug=slug,map_N=source.runtime.total,
            fingerprint=source.fingerprint,compensator_parameters=comp_params,student=False,
            checkpoint=scene['checkpoint']))
        factory=create_camera_preparer(MiniCam,variant='R')
        views=views_for(source,'final')
        camera=factory.from_source(source,views[0][1])
        renderer=create_renderer(camera,source.scene.background,source.runtime.total,use_graph=True)
        master,master_ids=source.runtime.full()
        def packet_for(cam,frame,condition):
            if condition=='full_direct':return master,master_ids,{}
            if condition=='full_copy':
                p,ids=source.runtime.full();return p,ids,{}
            return source.runtime.prepare(cam,ratio_budget=float(condition[5:]))
        # 数值与选点身份检查在计时之外，渲染器唯一oracle仍为原始A。
        for condition in CONDITIONS:
            for _,frame in views:
                old_camera=source.camera(frame);cam=factory.from_source(source,frame)
                p,ids,_=packet_for(cam,frame,condition)
                q,qids,_=packet_for(old_camera,frame,condition)
                assert torch.equal(ids,qids)
                ref=render_prepared(old_camera,q,source.scene.background,return_radii=True)
                image,radii=renderer.render(cam,p,source.scene.background)
                ok=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                assert ok,(slug,condition,frame)
                checks.append(dict(scene=scene['id'],condition=condition,frame=frame,K=len(ids),
                    ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),rgb_radii_bitwise=True))
        # 每种请求用所有18视角预热，容量/执行计划均已按地图上限初始化。
        for condition in CONDITIONS:
            for _,frame in views:
                cam=factory.from_source(source,frame);p,ids,_=packet_for(cam,frame,condition)
                renderer.render(cam,p,source.scene.background)
        torch.cuda.synchronize()
        monitor.check()
        for round_index in range(2):
            ordered_views=views if round_index==0 else list(reversed(views))
            for repeat in range(6):
                for index,(_,frame) in enumerate(ordered_views):
                    # 纯渲染输入在计时外重建，Full与LoD均使用相同G渲染器。
                    cam=factory.from_source(source,frame)
                    packets={condition:packet_for(cam,frame,condition) for condition in CONDITIONS}
                    for scope in ['raster','pipeline']:
                        for condition in rotate(CONDITIONS,index+repeat+round_index*3):
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            if scope=='pipeline':
                                current=factory.from_source(source,frame)
                                packet,ids,stats=packet_for(current,frame,condition)
                            else:
                                current=cam;packet,ids,stats=packets[condition]
                            image,radii=renderer.render(current,packet,source.scene.background)
                            torch.cuda.synchronize();ms=(time.perf_counter_ns()-tick)/1e6
                            samples.append(dict(scene=scene['id'],condition=condition,scope=scope,
                                round=round_index,repeat=repeat,frame=frame,N=source.runtime.total,K=len(ids),
                                wall_ms=ms,variant='R_teacher' if scope=='pipeline' else 'G_teacher'))
                    del packets
                monitor.check()
                print('PAIRED_PROGRESS',scene['id'],round_index,repeat,flush=True)
        write_csv(reports/'samples.csv',samples);write_csv(reports/'correctness.csv',checks)
        assert renderer.native.stats()[1:3]==[0,0],renderer.native.stats()
        del renderer,factory,source,model,master,master_ids,p,q,ref,image,radii,packet
        torch.cuda.empty_cache()
    monitor.close();write_json(reports/'GPU_OBSERVATIONS.json',monitor.rows)
    write_json(reports/'MODEL_IDENTITIES.json',identity)
    after={n:sha_file(n) for n in protected};changed=[n for n in protected if protected[n]!=after[n]]
    write_json(reports/'PROTECTED_AFTER.json',after)
    result=dict(passed=not changed,changed=changed,protected_files=len(protected),
        samples=len(samples),correctness_cases=len(checks),scenes=len(identity),external_active_samples=0,
        compensator='Teacher64/64',student=False,saved_render_images=0)
    write_json(reports/'VALIDATION.json',result);assert result['passed']
    print('PAIRED_COMPLETE',json.dumps(result),flush=True)


def read_json(p):return json.loads(Path(p).read_text())


if __name__=='__main__':main()
