"""绑定、13参数布局、真实选点、Teacher不变与学生普通计算对照。"""
from common import *
import gc
torch.set_num_threads(1);torch.set_num_interop_threads(1)
rows=[];models=[]
with torch.no_grad():
    for spec in SCENES:
        source=load_scene(spec['slug']);fingerprint=teacher_fingerprint(source['model'],source['pc']);student,payload=load_student(student_path(source),'cpu',fingerprint)
        old=native(source,'original_teacher');teacher=native(source,'teacher');engine=native(source,'student',diagnose=True)
        assert engine.compensator_info['width']==32 and len(student.deployment_parameters())==13
        record=source['records'][len(source['records'])//2];cam=camera(record)
        for condition in ['local0.1','local0.3','local0.5','local0.25','local0.01','local0.7','fixed0','global0.1','full']:
            a,ids,stats=request(old,cam,condition);b,bids,_=request(teacher,cam,condition);c,cids,info=request(engine,cam,condition)
            assert torch.equal(ids,bids) and torch.equal(ids,cids)
            for key,value in a['inputs'].items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,b['inputs'][key]);assert torch.isfinite(c['inputs'][key]).all()
            maximum=0.;residual_error=0.
            if len(ids) and condition!='full':
                ratio=float(condition[6:] if condition.startswith('global') else condition[5:]);truth,residual=ordinary_student(source,student,cam,ids,ratio,not condition.startswith('global'))
                actual=engine.native.diagnostics()[3];residual_error=float((actual-residual).abs().max())
                for key,value in truth['inputs'].items():
                    if isinstance(value,torch.Tensor) and value.numel():maximum=max(maximum,float((c['inputs'][key]-value).abs().max()))
            rows.append(dict(scene=spec['id'],condition=condition,K=len(ids),teacher_unchanged=True,same_ids=True,finite=True,max_attribute_abs=maximum,max_residual_abs=residual_error))
        models.append(dict(scene=spec['id'],student_checkpoint=str(student_path(source)),student_sha256=sha(student_path(source)),fingerprint=fingerprint,
            student_parameters=engine.compensator_info['parameters'],teacher_parameters=teacher.compensator_info['parameters'],student_workspace_bytes=engine.native.workspace_bytes(),teacher_workspace_bytes=teacher.native.workspace_bytes()))
        write_csv(HERE/'reports/PREFLIGHT_CHECKS.csv',rows);write_csv(HERE/'reports/MODELS.csv',models)
        del old,teacher,engine,source,student;gc.collect();print('STUDENT_PREFLIGHT',spec['id'],len(rows),flush=True)
assert not torch.cuda.is_initialized()
save(HERE/'reports/PREFLIGHT.json',dict(passed=True,scenes=len(models),checks=len(rows),teacher_unchanged=True,ids_identical=True,cuda_initialized=False,
    binary=extension().__file__,binary_sha256=sha(extension().__file__),max_residual_abs=max(x['max_residual_abs'] for x in rows),max_attribute_abs=max(x['max_attribute_abs'] for x in rows)))
