from common import *
import distill_compensation
from compensation_student import load_student
import csv


def invoke(arguments):
    sys.argv=['distill_compensation.py',*arguments]
    distill_compensation.main()


def main():
    output=ROOT/'models/functional_student_v2'
    invoke(['train',str(CHECKPOINTS['both_li']),'--dataset',DATASET,'--output',str(output),
        '--steps','24','--batch-points','256','--calibration-cases','12',
        '--image-every','12','--validate-every','12'])
    rows=list(csv.DictReader((output/'training.csv').open()))
    assert len(rows)==24 and any(float(r['image'])>0 for r in rows)
    student,state=load_student(output/'last.pt')
    assert state['step']==24 and len(state['optimizer']['state'])==len(list(student.parameters()))
    invoke(['train',str(CHECKPOINTS['both_li']),'--dataset',DATASET,'--output',str(output),
        '--steps','28','--batch-points','256','--calibration-cases','12',
        '--image-every','12','--validate-every','12','--resume',str(output/'last.pt')])
    rows=list(csv.DictReader((output/'training.csv').open()))
    assert len(rows)==28 and int(rows[-1]['step'])==28
    invoke(['evaluate',str(CHECKPOINTS['both_li']),'--dataset',DATASET,'--student',str(output/'best.pt'),
        '--output',str(ROOT/'reports/functional_quality_v2'),'--frames','2'])
    quality=json.loads((ROOT/'reports/functional_quality_v2/quality.json').read_text())
    save(ROOT/'reports/FUNCTIONAL_TRAINING.json',dict(passed=True,steps=28,resume_steps=4,
        gradient_image_steps=2,checkpoint_binding=True,quality_gates_passed=quality['all_quality_gates_passed'],
        full_distillation=False,performance_measured=False,images_saved=0))
    print('FUNCTIONAL_TRAINING_PASS',flush=True)


if __name__=='__main__':main()
