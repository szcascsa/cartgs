from pathlib import Path
import copy
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'source/python'))
sys.path.append('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python')
import torch
from compensation_student import StudentCompensator, distillation_loss, load_student, save_student
from spatial_compensation import SpatialCompensation


def make_case(teacher, n=128, view=True):
    x = torch.randn(n, 51).requires_grad_()
    ratio = torch.randn(1, 3).requires_grad_()
    condition = torch.randn(n, 8).requires_grad_() if view else None
    reference = (torch.rand(n, 1)*.2+.05).requires_grad_()
    value = teacher.encode(x) + teacher.ratio_projection(ratio)
    if view:
        value = value + teacher.view_projection(condition)
    hidden = torch.relu(teacher.body(torch.relu(value)))
    raw = torch.cat([head(hidden) for head in teacher.heads.values()], dim=1)
    target = torch.cat((raw[:, :3]*reference, raw[:, 3:]), dim=1)
    rotation = torch.randn(n, 4)
    rotation = (rotation/rotation.norm(dim=1, keepdim=True)).requires_grad_()
    master = [torch.randn(n, 3).requires_grad_(), torch.randn(n, 1).requires_grad_(),
        (torch.rand(n, 3)*.1+.001).requires_grad_(), rotation,
        torch.randn(n, 1, 3).requires_grad_(), torch.randn(n, 15, 3).requires_grad_()]
    return dict(inputs=x,ratio_code=ratio,condition=condition,reference=reference,target=target,
        master=master,softplus=True)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(131)
    teacher=SpatialCompensation(fourier_bands=0,ratio_feature_width=48,ratio_code=True,offset_only=True)
    teacher.enable_view_input()
    with torch.no_grad():
        teacher.view_projection.weight.normal_(0,.05)
        for head in teacher.heads.values():
            head[2].weight.normal_(0,.03)
            head[2].bias.normal_(0,.03)
    cases=[make_case(teacher) for _ in range(6)]
    student=StudentCompensator.from_teacher(teacher)
    assert sum(p.numel() for p in student.parameters())==3534
    student.teacher_fingerprint='a'*64
    student.calibrate(cases[:4])
    def loss(case):
        result=student(case['inputs'],case['ratio_code'],case['reference'],case['condition'])
        return sum(distillation_loss(student,case,result).values())
    before=float(loss(cases[4]))
    train_before=float(loss(cases[0]))
    optimizer=torch.optim.Adam(student.parameters(),lr=1e-3)
    for step in range(320):
        optimizer.zero_grad(set_to_none=True)
        value=loss(cases[step%4]);assert torch.isfinite(value)
        value.backward();optimizer.step()
    after=float(loss(cases[4]))
    train_after=float(loss(cases[0]))
    assert train_after<train_before*.9,(train_before,train_after)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(t.grad is None for c in cases for t in [c['inputs'],c['ratio_code'],c['reference'],c['condition'],*c['master']])
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'student.pt'
        save_student(path,student,metadata=dict(test=True),optimizer=optimizer,step=320,
            generator_state=torch.Generator().manual_seed(77).get_state())
        restored,payload=load_student(path,expected_fingerprint='a'*64)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in student.state_dict().items())
        try:load_student(path,expected_fingerprint='b'*64)
        except ValueError:pass
        else:raise AssertionError('Wrong teacher binding accepted')
        saved_opt=torch.optim.Adam(restored.parameters(),lr=1e-3);saved_opt.load_state_dict(payload['optimizer'])
        for model,opt in [(student,optimizer),(restored,saved_opt)]:
            opt.zero_grad(set_to_none=True)
            c=cases[0];v=model(c['inputs'],c['ratio_code'],c['reference'],c['condition'])
            sum(distillation_loss(model,c,v).values()).backward();opt.step()
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in student.state_dict().items())
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items():assert torch.equal(value,saved_opt.state_dict()['state'][key][name])
    # Antipodal quaternions represent the same rotation.
    with torch.no_grad():
        c=copy.copy(cases[0]);c['target']=cases[0]['target'].detach().clone()
        value=c['target'].clone();value[:,6:10]=-value[:,6:10];value[:,6]-=2
        assert float(distillation_loss(student,c,value)['rotation'])<1e-6
    for view in [False,True]:
        model=StudentCompensator(view)
        out=model(torch.empty(0,51),torch.zeros(1,3),torch.empty(0,1),torch.empty(0,8) if view else None)
        assert out.shape==(0,14)
    report=dict(passed=True,parameters=3534,initial_validation_loss=before,final_validation_loss=after,
        initial_training_loss=train_before,final_training_loss=train_after,
        synthetic_test=True,scene_quality_validated=False,
        teacher_gradients_absent=True,input_gradients_absent=True,checkpoint_roundtrip=True,
        optimizer_resume_bitwise=True,teacher_binding=True,quaternion_sign_equivalence=True,empty_input=True)
    (ROOT/'reports/CPU_STUDENT_TEST.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
