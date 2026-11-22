"""直接复用已验收冻结网络和原渲染入口；避免覆盖旧流程的common模块。"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math,os,statistics as st,subprocess,sys,time

HERE=Path(__file__).resolve().parent
PRIOR=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/research/compensator_distillation_replica_all_20261108')
sys.path.insert(0,str(PRIOR))
from scene_runtime import *
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
SCENES=json.loads((HERE/'reports/SCENES.json').read_text())
BY_CHECKPOINT={Path(s['checkpoint']).resolve():s for s in SCENES}
PROTOCOL=json.loads((HERE/'reports/PROTOCOL.json').read_text())

# 只替换冻结模型读入时的轨迹查表，网络与原数据流保持既有实现。
def trajectory_views(checkpoint,dataset):
    s=BY_CHECKPOINT[Path(checkpoint).resolve()]
    cameras=json.loads((Path(checkpoint)/'cameras.json').read_text())
    calibration={k:cameras[0][k] for k in ['width','height','fx','fy']}
    dataset=Path(dataset)
    if s['dataset_kind']=='Replica':
        images=sorted((dataset/'results').glob('frame*.jpg'));timestamps=np.array([float(p.stem[5:]) for p in images])
    else:
        entries=[line.split('#',1)[0].split() for line in (dataset/'rgb.txt').read_text().splitlines()]
        entries=[row for row in entries if row];assert all(len(row)==2 for row in entries)
        images=[dataset/r[1] for r in entries];timestamps=np.array([float(r[0]) for r in entries])
    records=np.atleast_2d(np.loadtxt(s['trajectory']));views=[];seen=set()
    for index,record in enumerate(records):
        image_index=int(np.abs(timestamps-record[0]).argmin())
        if abs(timestamps[image_index]-record[0])>.001:continue
        assert image_index not in seen;seen.add(image_index)
        pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(record[4:8]).as_matrix();pose[:3,3]=record[1:4]
        assert images[image_index].is_file()
        views.append(dict(name=str(images[image_index].relative_to(dataset)),timestamp=float(record[0]),
            pose=pose.tolist(),image=str(images[image_index]),trajectory_index=index,rgb_index=image_index))
    assert len(views)==len(records) and len(views)>30,(s['id'],len(views),len(records))
    return calibration,views

budget_viewer.load_views=trajectory_views


def load_scene(slug):
    scene=next(s for s in SCENES if s['slug']==slug)
    return scene,TeacherSamples(Path(scene['checkpoint']),Path(scene['dataset']))


def load_native(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def baseline_B():
    runtime=load_native('readonly_baseline_runtime',Path('/home/prmi/zwj/experiments/sparse_rasterizer_balanced_20261110/runtime.py'))
    native=load_native('sparse_rasterizer_balanced_v1',PROTOCOL['baseline_B']['path'])
    runtime.extension=lambda:native
    return runtime.SparseRenderer(4)


def bits_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False))


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def views_for(source,split='tune'):
    spec=PROTOCOL['tune_motion' if split=='tune' else 'final_motion'];n=len(source.scene.views)
    start=int(spec['continuous_fraction']*n)
    return [('continuous',(start+j)%n) for j in range(spec['continuous_frames'])]+[
        ('jump',int(x*n)%n) for x in spec['jump_fractions']]


@torch.no_grad()
def prepare(source,camera,frame,condition):
    if condition=='full':
        packet,ids=source.runtime.full();return packet,ids,dict(candidate_gaussians=len(ids))
    if condition.startswith('global'):
        ratio=float(condition[6:]);case=source.sample(frame,ratio,False)
        return case['packet'],case['ids'],dict(candidate_gaussians=source.runtime.total)
    if condition.startswith('fixed'):return source.runtime.prepare(camera,budget=int(condition[5:]))
    return source.runtime.prepare(camera,ratio_budget=float(condition[5:]))


def external_load(allowed=()):
    allowed=set(allowed)|{os.getpid()}
    raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
    active=[]
    for line in raw.splitlines():
        f=line.split()
        if len(f)>=7 and f[0].isdigit() and f[1].isdigit() and int(f[1]) not in allowed:
            if any(v.isdigit() and int(v)>0 for v in f[3:7]):active.append(line)
    return dict(time=time.time(),active=active,pmon=raw)
