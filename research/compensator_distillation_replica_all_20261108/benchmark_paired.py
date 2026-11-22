"""冻结Teacher/Student的同后端配对测速；不修改模型，不保存渲染图片。"""
from scene_runtime import *
import argparse
import csv
import subprocess
import statistics
from compensation_student import load_student
from deployment_runtime import DeploymentRenderer
from distill_compensation import ratio_controls


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def process_identities():
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    result = {}
    for line in raw.splitlines():
        if line.strip() and int(line)!=os.getpid():
            pid = int(line)
            fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
            result[str(pid)] = int(fields[19])
    return result


def guard(allowed, observations):
    actual = process_identities()
    assert all(allowed.get(pid)==start for pid,start in actual.items()), ('New GPU process',actual)
    raw = subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
    for line in raw.splitlines():
        if line.strip() and not line.startswith('#'):
            fields = line.split()
            if len(fields)>4 and fields[1] in actual:
                assert all(v=='-' or int(v)==0 for v in fields[3:7]), ('Resident GPU became active',line)
    state = subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw',
        '--format=csv,noheader'],text=True).strip()
    observations.append(dict(time=time.time(),foreign_processes=actual,pmon=raw,gpu=state))


def order(index):
    return ['teacher','student'] if index%4 in (0,3) else ['student','teacher']


@torch.no_grad()
def benchmark_scene(slug, output, allowed):
    scene, source = load_scene(slug)
    student_path = ROOT/'models'/slug/'continuous_calibration/best.pt'
    student,_ = load_student(student_path,'cuda',source.fingerprint)
    engines = dict(teacher=source.runtime,student=DeploymentRenderer(source.scene.model,
        source.scene.gaussians,source.scene.config,student=student))
    observations = []
    guard(allowed,observations)
    conditions = [('1000',dict(budget=1000)),('5000',dict(budget=5000)),
        ('1pct',dict(ratio_budget=.01)),('10pct',dict(ratio_budget=.1)),('70pct',dict(ratio_budget=.7))]
    frames = [(i*73+19)%len(source.scene.views) for i in range(80)]
    cameras = [source.camera(frame) for frame in frames]
    warm_end = time.perf_counter()+2
    i = 0
    while time.perf_counter()<warm_end:
        for engine in engines.values():
            packet,_,_ = engine.prepare(cameras[i%80],budget=5000)
            render_prepared(cameras[i%80],packet,source.scene.background)
        torch.cuda.synchronize()
        i += 1
    rows = []
    started = time.time()
    for repeat in range(3):
        sequence = conditions if repeat%2==0 else list(reversed(conditions))
        for condition,request in sequence:
            guard(allowed,observations)
            for i,(frame,cam) in enumerate(zip(frames,cameras)):
                ids_pair = []
                for name in order(i+repeat):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    packet,ids,stats = engines[name].prepare(cam,**request)
                    torch.cuda.synchronize()
                    prepare_ms = (time.perf_counter()-begin)*1000
                    render_prepared(cam,packet,source.scene.background)
                    torch.cuda.synchronize()
                    total_ms = (time.perf_counter()-begin)*1000
                    ids_pair.append(ids)
                    if i>=20:
                        rows.append(dict(scene=scene['id'],repeat=repeat,condition=condition,
                            version=name,frame=frame,N=stats['candidate_gaussians'],K=len(ids),
                            prepare_ms=prepare_ms,total_ms=total_ms,
                            graph=stats['graph_replay'],device_counts=stats['device_counts']))
                assert torch.equal(*ids_pair), (slug,frame,condition,'Selection mismatch')
            print('SPEED_PREPARE',slug,repeat,condition,flush=True)
    write_csv(output/'prepare_samples.csv',rows)
    # 单独测量已有tiled CUDA入口，含Feature查询/网络/解码/输出分配，排除Selector和Rasterizer。
    component = []
    b = source.runtime._base
    for frame in [19,len(source.scene.views)//2,len(source.scene.views)-20]:
        cam = source.camera(frame)
        for ratio in [.01,.05,.1,.3,.5,.7,.2]:
            for local in [False,True]:
                case = source.sample(frame,ratio,local)
                view = b.empty
                if local:
                    view = source.runtime._ext.view(b.anchors.centers,b.view_centers4,b.view_corners4,
                        cam.camera_center.contiguous(),cam.world_view_transform,cam.full_proj_transform,
                        b.model.selector_view_scene_scale,cam.image_width/cam.image_height,cam.znear)
                weights,code = ratio_controls(ratio,source.runtime.knots,b.spatial.ratio_encoding_values)
                args = (b.offset,case['ids'],b.owners,b.reference,b.tables)
                calls = {name:(lambda engine=engine:engine._ext._compensate_tiled(*args,
                    engine._compensator,b.compensated_master,view,engine._view_weight,weights,code,
                    local,b.joint_schema,case['softplus'],16)) for name,engine in engines.items()}
                for call in calls.values():
                    for _ in range(10): result=call()
                    assert int(result[2])==0
                for repeat in range(3):
                    for name in order(repeat):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start.record()
                        for _ in range(30): result=calls[name]()
                        end.record();end.synchronize()
                        assert int(result[2])==0
                        component.append(dict(scene=scene['id'],frame=frame,ratio=ratio,local=local,
                            K=len(case['ids']),repeat=repeat,version=name,cuda_ms=start.elapsed_time(end)/30,
                            tile=16,launches_per_sample=30))
        guard(allowed,observations)
        print('SPEED_COMPONENT',slug,frame,flush=True)
    write_csv(output/'component_samples.csv',component)
    summary = []
    for condition,_ in conditions:
        group = [r for r in rows if r['condition']==condition]
        item = dict(condition=condition,N_median=statistics.median(r['N'] for r in group),
            K_median=statistics.median(r['K'] for r in group),K_min=min(r['K'] for r in group),K_max=max(r['K'] for r in group))
        for metric in ['prepare_ms','total_ms']:
            medians = {name:statistics.median(r[metric] for r in group if r['version']==name) for name in engines}
            item[metric] = dict(**medians,speedup=medians['teacher']/medians['student'],
                reduction_pct=100*(1-medians['student']/medians['teacher']),saved_ms=medians['teacher']-medians['student'],
                repeat_speedups=[statistics.median(r[metric] for r in group if r['version']=='teacher' and r['repeat']==j)/
                    statistics.median(r[metric] for r in group if r['version']=='student' and r['repeat']==j) for j in range(3)])
        summary.append(item)
    comp_summary = []
    for local in [False,True]:
        for ratio in [.01,.05,.1,.3,.5,.7,.2]:
            group = [r for r in component if r['local']==local and r['ratio']==ratio]
            medians = {name:statistics.median(r['cuda_ms'] for r in group if r['version']==name) for name in engines}
            comp_summary.append(dict(local=local,ratio=ratio,K_median=statistics.median(r['K'] for r in group),
                **medians,speedup=medians['teacher']/medians['student']))
    result = dict(scene=scene['id'],map_N=scene['gaussians'],checkpoint=scene['checkpoint'],
        student_checkpoint=str(student_path),student_sha256=sha(student_path),teacher_fingerprint=source.fingerprint,
        backend='same v31 DeploymentRenderer; original Teacher vs final Student',started=started,ended=time.time(),
        selection_ids_identical=True,prepare_samples=len(rows),component_samples=len(component),
        warmup_per_condition=20,measured_views_per_condition_repeat=60,repeats=3,
        prepare_includes_final_cuda_sync=True,component_tile=16,
        component_scope='CUDA Event batch: gather+residual+MLP+decode+publish; allocation/launch GPU gaps may be included',
        resident_process_identity=allowed,load_observations=observations,
        prepare=summary,component=comp_summary,render_images_saved=0)
    save(output/'summary.json',result)
    print('SPEED_SCENE_COMPLETE',slug,json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    allowed = process_identities()
    observations=[]
    guard(allowed,observations)
    # 初始自身尚未建立GPU工作，验证全卡无实际工作；驻留进程由pmon核查。
    util=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip())
    assert util<=5,('GPU busy before benchmark',util)
    slugs=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
    save(args.output/'PROTOCOL.json',dict(slugs=slugs,resident_processes=allowed,initial_observations=observations,
        ratios=[.01,.05,.1,.3,.5,.7,.2],same_backend=True,train_changes=False,initial_gpu_util=util,
        component_tile=16,component_cuda_event=True,selection_check_outside_timing=True))
    for slug in slugs:
        p=args.output/slug;p.mkdir()
        benchmark_scene(slug,p,allowed)
        torch.cuda.empty_cache()
    save(args.output/'COMPLETE.json',dict(completed=slugs,passed=True,time=time.time()))


if __name__=='__main__':
    main()
