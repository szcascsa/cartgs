from common import *
import distill_compensation


def main():
    output=ROOT/'models/fr1_student32_image_calibration'
    sys.argv=['distill_compensation.py','evaluate',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
        '--student',str(ROOT/'models/fr1_student32/best.pt'),
        '--output',str(ROOT/'reports/fr1_quality_independent'),'--frames','16']
    distill_compensation.main()
    sys.argv=['distill_compensation.py','train',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
        '--output',str(output),'--steps','2000','--initialize',str(ROOT/'models/fr1_student32/best.pt')]
    distill_compensation.main()
    sys.argv=['distill_compensation.py','evaluate',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
        '--student',str(output/'best.pt'),'--output',str(ROOT/'reports/fr1_image_quality'),'--frames','16']
    distill_compensation.main()
    save(ROOT/'reports/IMAGE_CALIBRATION_COMPLETE.json',dict(passed=True,additional_steps=2000,
        checkpoint=str(output/'best.pt'),quality=str(ROOT/'reports/fr1_image_quality/quality.json'),
        speed_measured=False,images_saved=0))


if __name__=='__main__':main()
