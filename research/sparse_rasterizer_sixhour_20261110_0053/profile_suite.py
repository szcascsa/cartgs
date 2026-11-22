"""冻结二进制的独立Nsight归因；不将Profiling数据并入正式延迟。"""
from pathlib import Path
import argparse,json,subprocess,time

HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'
NSYS='/usr/local/cuda-11.8/bin/nsys'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    cases=[(variant,condition,'raster') for variant in ['A','B','C'] for condition in ['local0.01','full']]
    cases += [('G','full','raster'),('G','local0.01','pipeline')]
    rows=[]
    for variant,condition,scope in cases:
        label=f'{args.output}_{variant}_{scope}_{condition.replace(".","p")}'
        trace=HERE/'evaluation'/label
        command=[NSYS,'profile','--trace=cuda,nvtx','--cuda-graph-trace=node',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false',
            '-o',str(trace),PYTHON,'-u',str(HERE/'profile_frozen.py'),'--output',label,
            '--backend',variant,'--condition',condition,'--scope',scope]
        tick=time.time()
        with (output/(label+'.log')).open('x') as log:
            subprocess.run(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        with (output/(label+'_stats.log')).open('x') as log:
            subprocess.run([NSYS,'stats','--report','gpukernsum,cudaapisum,gpumemtimesum',
                '--format','csv','--output',str(output/label),str(trace)+'.nsys-rep'],
                cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows.append(dict(variant=variant,condition=condition,scope=scope,calls=20,
            trace=str(trace)+'.nsys-rep',elapsed_s=time.time()-tick,label=label))
        print('PROFILE_CASE_COMPLETE',label,flush=True)
    (output/'SUMMARY.json').write_text(json.dumps(dict(passed=True,rows=rows,
        timing_use='仅Kernel和API归因；不计入正式配对墙钟结果'),ensure_ascii=False,indent=2))
    print('PROFILE_SUITE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
