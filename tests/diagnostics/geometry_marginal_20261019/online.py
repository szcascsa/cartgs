"""实际全序列SLAM；Shadow gate后配对两场景两seed，所有质量最后统一评价。"""
import os,subprocess,json,time,signal,re,resource,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;R=Path('/home/prmi/zwj/mycartgs');PY='/home/prmi/anaconda3/envs/zwj/bin/python'
def read(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
P=read(ROOT/'online_protocol.json');D={d['scene']:d for d in P['datasets']}
ENV=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'runtime/python'),LD_LIBRARY_PATH=str(R/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
def execute(command,directory,env,kind):
    directory.mkdir(parents=True,exist_ok=True);path=directory/(kind+'_status.json')
    if path.exists():
        old=read(path);assert old['state']=='success',old;return old
    start=time.monotonic();monitor=[]
    with open(directory/(kind+'.log'),'wb')as log:
        child=subprocess.Popen(list(map(str,command)),cwd=R,env=env,stdout=log,stderr=log,start_new_session=True)
        try:
            while child.poll()is None:
                assert time.monotonic()-start<7200,'wall timeout'
                status=dict(state='running',pid=child.pid,directory=str(directory),kind=kind,elapsed_s=time.monotonic()-start);save(path,status);save(ROOT/'online_status.json',status)
                try:
                    text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip();gpu,mem=map(float,text.split(','));vm=Path(f'/proc/{child.pid}/status').read_text();match=re.search(r'VmRSS:\s+(\d+)',vm)
                    if match is not None:
                        rss=int(match.group(1));stat=Path(f'/proc/{child.pid}/stat').read_text().split();ticks=int(stat[13])+int(stat[14]);monitor.append(dict(elapsed_s=time.monotonic()-start,gpu_percent=gpu,vram_mib=mem,rss_kib=rss,cpu_ticks=ticks))
                        if len(monitor)%10==0:save(directory/(kind+'_resources.json'),monitor)
                except(FileNotFoundError,ProcessLookupError):pass
                time.sleep(1)
        except BaseException:os.killpg(child.pid,signal.SIGTERM);child.wait();raise
        result=dict(state='success'if child.wait()==0 else'failed',exit_code=child.returncode,elapsed_s=time.monotonic()-start,command=list(map(str,command)),peak_ram_kib=max((x['rss_kib']for x in monitor),default=0),peak_vram_mib=max((x['vram_mib']for x in monitor),default=0))
    save(path,result);save(directory/(kind+'_resources.json'),monitor);assert result['state']=='success',str(directory/(kind+'.log'));return result
def train(scene,seed,arm):
    d=D[scene];directory=ROOT/'online_runs'/f'{scene}_{seed}_{arm}';output=directory/'model';env=dict(ENV,CARTGS_AUDIT_SEED=str(seed),CARTGS_MAPPING_TRACE=str(directory/'mapping_trace.csv'))
    cmd=[R/'bin/tum_rgbd',R/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',d['orb'],d['mapper'],d['data'],Path(d['data'])/'associations.txt',output,'no_viewer','--selector-config',ROOT/'configs'/(scene+'_'+arm+'.yaml')]
    result=execute(cmd,directory,env,'training');endpoints=list(output.glob('*_shutdown'));assert len(endpoints)==1;checkpoint=endpoints[0];assert int(checkpoint.name.split('_')[0])==d['target_mapping_iterations']
    state=read(checkpoint/'ply/mapping_stage_state.json');assert state['online']['mappings']+state['tail']['mappings']==d['target_mapping_iterations'];assert state['online']['lod_nonfinite']==state['tail']['lod_nonfinite']==0
    import torch
    model=torch.jit.load(str(checkpoint/'ply/selector.pt'),map_location='cpu').state_dict();assert model['soft_net.weight'].shape==(2,128)
    assert all(torch.isfinite(v).all()for v in model.values())
    cloud=checkpoint/'ply/point_cloud'/('iteration_'+str(d['target_mapping_iterations']))/'point_cloud.ply';header=cloud.open('rb').read(8192).split(b'end_header')[0].decode();n=next(int(s.split()[-1])for s in header.splitlines()if s.startswith('element vertex '))
    row=dict(scene=scene,seed=seed,arm=arm,model=str(output),checkpoint=str(checkpoint),N=n,training=result,state=state)
    print('TRAINING_VALID',scene,seed,arm,n,flush=True);return row
def main():
    assert read(ROOT/'optimized_exact_validation.json')['all_passed']
    while not(ROOT/'optimized_render_validation.json').exists():time.sleep(2)
    assert read(ROOT/'optimized_render_validation.json')['all_passed']
    shadow=train('fr1_room',2026,'shadow');log=(Path(shadow['model']).parent/'training.log').read_text(errors='replace');passed=[int(s)for s in re.findall(r'event=shadow_exact iteration=(\d+)',log)];assert any(i<=2000 for i in passed)and any(7000<=i<=10000 for i in passed)and any(i>=15000 for i in passed),passed
    save(ROOT/'shadow_validation.json',dict(all_passed=True,reference_exact_iterations=passed,raw_teacher_only=True,model=shadow))
    models=[]
    for item in P['order']:
        models.append(train(**item));save(ROOT/'online_training_results.json',models)
    from evo.tools import file_interface
    from evo.core import sync
    plans={}
    for scene,d in D.items():
        group=[z for z in models if z['scene']==scene];rgb=np.array([float(s.split()[0])for s in(Path(d['data'])/'rgb.txt').read_text().splitlines()if s.strip()and not s.startswith('#')]);sets=[]
        for row in group:
            ref=file_interface.read_tum_trajectory_file(str(Path(d['data'])/'groundtruth.txt'));est=file_interface.read_tum_trajectory_file(str(Path(row['model'])/'CameraTrajectory_TUM.txt'));_,est=sync.associate_trajectories(ref,est,max_diff=.08);indices=[int(np.abs(rgb-t).argmin())for t in est.timestamps];unique={i for i in indices if indices.count(i)==1};sets.append(unique)
        common=sorted(set.intersection(*sets));assert len(common)>=8;chosen=[common[int(i)]for i in np.linspace(0,len(common)-1,min(P['evaluation_views'],len(common)),dtype=int)];save(ROOT/(scene+'_common_frames.json'),dict(gt_indices=chosen,available_common=len(common),policy=P['evaluation_frames']));plans[scene]=dict(reference_N=min(z['N']for z in group),frames=len(chosen),available_common=len(common))
    save(ROOT/'evaluation_plan.json',plans);results=[]
    for row in models:
        for policy,ratio in [('full',None),('production',.01),('production',.05),('same_k',.01),('same_k',.05)]:
            directory=Path(row['model']).parent/'evaluation'/(policy+'_'+str(ratio));cmd=[PY,ROOT/'evaluate_online.py',row['model'],D[row['scene']]['data'],'--output_path',directory,'--shutdown-model',Path(row['checkpoint']).name,'--no-save-images','--evaluation-frame-set',ROOT/(row['scene']+'_common_frames.json'),'--audit-policy',policy]
            if ratio is not None:cmd+=['--selector',Path(row['checkpoint'])/'ply/selector.pt','--selector-ratio',str(ratio),'--selector-config',ROOT/'configs'/(row['scene']+'_'+row['arm']+'.yaml')]
            elapsed=execute(cmd,directory,ENV,'eval');metrics={k:float(v)for k,v in(s.split(':',1)for s in(directory/'eval.txt').read_text().splitlines())};entry=dict(scene=row['scene'],seed=row['seed'],arm=row['arm'],policy=policy,ratio=ratio,metrics=metrics,ate=read(directory/'ate_standard.json'),selection=read(directory/'selector_stats.json'),output=str(directory));results.append(entry);save(ROOT/'online_evaluation_results.json',results);print('QUALITY_VALID',row['scene'],row['seed'],row['arm'],policy,ratio,metrics['psnr'],flush=True)
    save(ROOT/'online_complete.json',dict(all_passed=True,models=len(models),quality_cases=len(results),shadow_gate=True))
if __name__=='__main__':
    try:main()
    except BaseException:save(ROOT/'online_failure.json',dict(traceback=traceback.format_exc()));raise
