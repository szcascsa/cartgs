"""错误绑定拒绝、CPU独立加载、输入隔离和两条学生入口一致。"""
from common import *
import gc
torch.set_num_threads(1);torch.set_num_interop_threads(1)
checks=[]
with torch.no_grad():
    for slug in ['Replica__office2__rgbd','Replica__room2__mono']:
        source=load_scene(slug);path=student_path(source);noise_path=ORIGINAL/'noise_snapshots'/(slug+'_seed0.pt')
        noise=torch.load(noise_path,map_location='cpu',weights_only=True);noise=noise['noise'] if isinstance(noise,dict) else noise;source['reference'].noise.copy_(noise)
        fingerprint=teacher_fingerprint(source['model'],source['pc']);student,_=load_student(path,'cpu',fingerprint)
        loaded=CpuRenderer.from_checkpoint(source['checkpoint'],noise_path,threads=6,student_checkpoint=path)
        direct=CpuRenderer(source['model'],source['pc'],noise,6,student=student);cam=camera(source['records'][len(source['records'])//2])
        for condition in ['local0.1','local0.25','local0.5','fixed0','fixed1000','global0.1','full']:
            a,ids,_=request(loaded,cam,condition);b,other,_=request(direct,cam,condition);assert torch.equal(ids,other)
            for key,value in a['inputs'].items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,b['inputs'][key])
            checks.append(dict(scene=slug,test='CPU_checkpoint_vs_instance',condition=condition,passed=True))
        wrong=FROZEN/'models'/'Replica__office0__mono'/'continuous_calibration/best.pt'
        try:CpuRenderer(source['model'],source['pc'],noise,6,student_checkpoint=wrong)
        except ValueError:checks.append(dict(scene=slug,test='Wrong_teacher_binding_rejected',condition='',passed=True))
        else:raise AssertionError('Accepted a student from another map')
        student.teacher_fingerprint='0'*64
        try:CpuRenderer(source['model'],source['pc'],noise,6,student=student)
        except ValueError:checks.append(dict(scene=slug,test='Wrong_instance_binding_rejected',condition='',passed=True))
        else:raise AssertionError('Accepted a mismatched student instance')
        # Cloned snapshot is independent of later changes to the supplied student.
        student.teacher_fingerprint=fingerprint;a,ids,_=request(direct,cam,'local0.3')
        for v in student.parameters():v.add_(1.)
        b,other,_=request(direct,cam,'local0.3');assert torch.equal(ids,other)
        for key,value in a['inputs'].items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,b['inputs'][key])
        checks.append(dict(scene=slug,test='Student_source_snapshot_isolated',condition='',passed=True))
        del source,student,loaded,direct;gc.collect()
assert not torch.cuda.is_initialized()
write_csv(HERE/'reports/BINDING_CHECKS.csv',checks);save(HERE/'reports/BINDING_TESTS.json',dict(passed=True,checks=len(checks),cuda_initialized=False))
print('STUDENT_BINDING_PASS',len(checks),flush=True)
