import argparse,gc,json
from pathlib import Path
import torch
from af_runtime import ROOT,Study,train,save
parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True);a=parser.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
queue=json.loads(Path(a.queue).read_text());study=Study(queue.get('scene','fr1_xyz'));runs=[]
for job in queue['jobs']:
    runs.append(train(study,job,queue['tier']));save(ROOT/(Path(a.queue).stem+'_results.json'),runs)
    gc.collect();torch.cuda.empty_cache()
save(ROOT/(Path(a.queue).stem+'_complete.json'),dict(runs=len(runs)))
