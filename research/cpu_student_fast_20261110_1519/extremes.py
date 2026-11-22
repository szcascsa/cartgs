"""私有合成学生覆盖Softplus/Opacity/旋转边界与SIMD尾部，不写原模型。"""
from common import *
import gc
torch.set_num_threads(1);torch.set_num_interop_threads(1)
source=load_scene('Replica__office2__rgbd');model,pc=source['model'],source['pc'];noise=source['reference'].noise;fingerprint=teacher_fingerprint(model,pc)
student,_=load_student(student_path(source),'cpu',fingerprint);cam=camera(source['records'][len(source['records'])//2]);checks=[]
with torch.no_grad():
    for case,value in enumerate([-120.,-100.,-80.,-30.,-20.,-12.,-1.,0.,5.,19.458675,20.,50.,100.]):
        test=copy.deepcopy(student)
        for v in test.parameters():v.zero_()
        test.heads['scale'].bias.fill_(value);test.heads['other'].bias[0]=value
        test.heads['rotation'].bias.copy_(torch.tensor([-1. if case%3==0 else .01,1e-12 if case%3==0 else .02,-.04,.001]))
        a=Baseline(model,pc,noise,6,student=test);b=CpuRenderer(model,pc,noise,6,student=test,student_kernel=2,student_decode=1,student_context=1)
        for K in [0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33]:
            old_error=new_error=None
            try:old,ids,_=a.prepare(cam,budget=K)
            except RuntimeError as error:old_error=type(error).__name__
            try:new,selected,_=b.prepare(cam,budget=K)
            except RuntimeError as error:new_error=type(error).__name__
            assert old_error==new_error,(value,K,old_error,new_error)
            if old_error:
                checks.append(dict(case=case,value=value,K=K,node='expected_rejection',max_abs=0.,max_relative=0.,finite=True));continue
            assert torch.equal(ids,selected)
            for name,v in new['inputs'].items():
                if isinstance(v,torch.Tensor):
                    assert torch.isfinite(v).all();reference=old['inputs'][name];error=(v-reference).abs();denominator=reference.abs().clamp_min(1e-30)
                    checks.append(dict(case=case,value=value,K=K,node=name,max_abs=float(error.max()) if v.numel() else 0.,max_relative=float((error/denominator).max()) if v.numel() else 0.,finite=True))
        del a,b,test;gc.collect();print('STUDENT_FAST_EXTREME',value,len(checks),flush=True)
write_csv(HERE/'reports/EXTREME_CHECKS.csv',checks);save(HERE/'reports/EXTREMES.json',dict(passed=True,checks=len(checks),original_weights_unchanged=True,cuda_initialized=torch.cuda.is_initialized()))
