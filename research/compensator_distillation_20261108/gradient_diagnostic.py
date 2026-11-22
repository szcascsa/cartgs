from common import *
from compensation_student import load_student,distillation_loss
from distill_compensation import TeacherSamples,student_residual,render_student


def flatten_gradients(gradients,parameters):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
        for g,p in zip(gradients,parameters)])


def main():
    source=TeacherSamples(CHECKPOINTS['both_li'],DATASET)
    student,_=load_student(ROOT/'models/fr1_student32/best.pt','cuda',source.fingerprint)
    parameters=list(student.parameters());rows=[]
    for ratio in [.01,.05,.1,.3,.5,.7]:
        for local in [False,True]:
            case=source.sample(395,ratio,local)
            output=student_residual(student,case)
            loss=sum(distillation_loss(student,case,output).values())
            attribute=flatten_gradients(torch.autograd.grad(loss,parameters,allow_unused=True),parameters)
            image=render_student(student,case,source.scene.background,source.runtime._base.pc.active_sh_degree)
            with torch.no_grad():target=render_prepared(case['camera'],case['packet'],source.scene.background)['render']
            photo_loss=(image-target).square().mean()
            photo=flatten_gradients(torch.autograd.grad(photo_loss,parameters,allow_unused=True),parameters)
            a=float(attribute.norm());p=float(photo.norm())
            rows.append(dict(ratio=ratio,local=local,K=len(case['ids']),attribute_loss=float(loss),
                photo_loss=float(photo_loss),attribute_gradient_norm=a,photo_gradient_norm=p,
                cosine=float(torch.dot(attribute,photo)/(attribute.norm()*photo.norm()).clamp_min(1e-20)),
                photo_to_attribute_gradient=p/max(a,1e-20),
                nominal_frequency_adjusted_ratio=p/max(a,1e-20)/20))
            print('GRAD',ratio,local,a,p,flush=True)
    save(ROOT/'reports/DISTILLATION_GRADIENTS.json',dict(rows=rows,diagnostic_only=True,
        parameter_updates=0,view=395,image_weight=1,image_every=20))


if __name__=='__main__':main()
