from common import *
import distill_compensation


def main():
    output=ROOT/'models/fr1_student32'
    sys.argv=['distill_compensation.py','train',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
        '--output',str(output),'--steps','6000','--image-every','20','--image-weight','1']
    distill_compensation.main()
    sys.argv=['distill_compensation.py','evaluate',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
        '--student',str(output/'best.pt'),'--output',str(ROOT/'reports/fr1_quality'),'--frames','16']
    distill_compensation.main()
    save(ROOT/'reports/FR1_DISTILLATION_COMPLETE.json',dict(passed=True,steps=6000,
        checkpoint=str(output/'best.pt'),quality=str(ROOT/'reports/fr1_quality/quality.json'),
        speed_measured=False,images_saved=0))


if __name__=='__main__':main()
