"""队列启动前验证原生条件路由和LI隔离，不重跑历史实验。"""
from pathlib import Path
import json
import subprocess
import time
from run_batch import R,environment
from pipeline import write,read


def main():
    folder=R/'validation'/('native_'+str(time.time_ns()));folder.mkdir(parents=True)
    tests=[('joint_view_conditioning_test',m) for m in ['unit','cross_unit','view_li','view_li_empty','view_li_isolation']]
    tests += [('local_importance_test',m) for m in ['unit','collection','isolation']]
    tests += [('lod_health_guard_test','')]
    result=[]
    for binary,mode in tests:
        name=binary+'_'+mode;command=[str(R/'runtime/bin'/binary)]
        if mode:command+=[mode,str(folder/name)]
        with (folder/(name+'.log')).open('w') as log:
            p=subprocess.run(command,cwd=R,env=environment(),stdout=log,stderr=subprocess.STDOUT)
        assert p.returncode==0,(binary,mode,p.returncode)
        result.append(dict(binary=binary,mode=mode,passed=True))
        write(R/'reports/STARTUP_VALIDATION.json',dict(state='running',tests=result))
    # Selection hashes replace large per-frame ID exports, without changing any rendering.
    script=R/'source/python/run.py';text=script.read_text()
    old='selected_rows.append(selected_ids.cpu().numpy().astype(np.int32,copy=False))';assert text.count(old)==1
    text=text.replace(old,"selected_rows.append(__import__('hashlib').sha256(selected_ids.cpu().numpy().tobytes()).hexdigest())")
    a=text.index('        offsets=np.concatenate(([0],np.cumsum(');b=text.index('        aggregate_keys =',a)
    text=text[:a]+"        with open(os.path.join(output_result_path,'selection_hashes.json'),'w') as stream:\n            json.dump(selected_rows,stream)\n"+text[b:]
    (R/'source/python/run_metrics_only.py').write_text(text)
    write(R/'reports/STARTUP_VALIDATION.json',dict(state='passed',passed=True,tests=result,render_images_saved=0))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
