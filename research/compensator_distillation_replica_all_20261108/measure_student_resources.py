"""Teacher与最终蒸馏学生的独立进程显存配对；不训练或修改权重。"""
from scene_runtime import *
import argparse
import subprocess
from measure_lod_memory import worker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--slug');parser.add_argument('--mode');parser.add_argument('--version')
    args=parser.parse_args()
    if args.slug:
        path=ROOT/'models'/args.slug/'continuous_calibration/best.pt'
        before=sha(path)
        worker(args.slug,args.mode,args.output,student_checkpoint=path if args.version=='student' else None)
        assert sha(path)==before
        return
    args.output.mkdir(parents=True,exist_ok=False)
    modes=['runtime_full','0.01','0.05','0.1','0.3','0.5','0.7','runtime_switch']
    for slug in ['Replica__office2__rgbd','Replica__room0__rgbd']:
        for i,mode in enumerate(modes):
            for version in (['teacher','student'] if i%2==0 else ['student','teacher']):
                p=args.output/f'{slug}__{version}__{mode}.json'
                with p.with_suffix('.log').open('x') as f:
                    code=subprocess.call([sys.executable,str(Path(__file__).resolve()),'--slug',slug,
                        '--mode',mode,'--version',version,'--output',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                assert code==0,p.with_suffix('.log').read_text()[-2000:]
                print('STUDENT_MEMORY_CASE_DONE',slug,version,mode,flush=True)
            a,b=[json.loads((args.output/f'{slug}__{v}__{mode}.json').read_text()) for v in ['teacher','student']]
            assert len(a['rows'])==len(b['rows'])
            assert all(all(x[k]==y[k] for k in ['mode','cycle','frame','K','selection_sha256'])
                for x,y in zip(a['rows'],b['rows']))
    save(args.output/'COMPLETE.json',dict(passed=True,workers=32,selection_ids_identical=True,
        teacher_frozen=True,student_stage='continuous_calibration/best.pt',training=False,
        models_changed=False,saved_images=0))


if __name__=='__main__':main()
