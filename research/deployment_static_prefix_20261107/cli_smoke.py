from common import *
import deployment_infer
import csv


def main():
    output=ROOT/'evaluation/CLI_SMOKE.csv'
    sys.argv=['deployment_infer.py',str(CHECKPOINTS['both_li']),'--dataset',DATASET,
              '--budget','1000','--frames','6','--output',str(output)]
    deployment_infer.main()
    rows=list(csv.DictReader(output.open()))
    assert len(rows)==6
    assert all(int(row['K'])==min(1000,int(row['candidate_N'])) for row in rows)
    assert all(np.isfinite(float(row['prepare_render_ms'])) for row in rows)
    save(ROOT/'reports/CLI_SMOKE.json',dict(passed=True,frames=len(rows),images_saved=0,output=str(output)))


if __name__=='__main__':main()
