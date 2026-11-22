"""正式计时、Profiler、生命周期和真实SLAM分开；外部GPU进程只记录。"""
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');S=R/'source';B=R/'build'
CP=Path('/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply')
DATA='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz'
F='/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1'
ENV=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',CARTGS_AUDIT_SEED='2026',
    LD_LIBRARY_PATH=str(S/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
for key in ['CARTGS_COMPENSATION_SPLIT4','CARTGS_BUDGET_METHOD','CARTGS_BUDGET_AUDIT','CARTGS_MLP_GRAD_AUDIT','CARTGS_MAPPING_TRACE']:
    ENV.pop(key,None)
samples=[];runs=[]
def run(name,cmd,env=ENV):
    started=time.monotonic();run_dir=R/name;run_dir.mkdir(exist_ok=True)
    (run_dir/'command.json').write_text(json.dumps(list(map(str,cmd)),indent=2))
    with (run_dir/'run.log').open('w') as log:
        process=subprocess.Popen(list(map(str,cmd)),cwd=S,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        while process.poll() is None:
            def query(kind,fields):
                return subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(run=name,seconds=time.monotonic()-started,pid=process.pid,
                gpu=query('gpu','utilization.gpu,memory.used,clocks.sm,power.draw'),apps=query('compute-apps','pid,process_name,used_memory')))
            time.sleep(1)
    row=dict(name=name,wall_s=time.monotonic()-started,returncode=process.returncode)
    runs.append(row);(R/'validation_runs.json').write_text(json.dumps(runs,indent=2));(R/'gpu_samples.json').write_text(json.dumps(samples,indent=2))
    assert process.returncode==0,row
    print('PASS',row,flush=True)

if __name__=='__main__':
    # 正在进行的首次生产编译完成后再启动本脚本。
    run('build_final',['/home/prmi/.local/bin/cmake','--build',B,'-j2','--target',*json.loads((R/'targets.json').read_text())])
    run('phase1_final',[S/'bin/fused_mapping_adam_test',F,R/'phase1_final'])
    run('phase2_final',[S/'bin/fused_compensation_decode_test',F,R/'phase2_final'])
    run('phase3_final',[S/'bin/fused_anchor_input_test',CP,F,R/'phase3_final'])
    for target,args in [('private_compensation_cuda_test',['cuda','fused']),('private_compensation_mapping_test',['cuda']),
                        ('gaussian_split_sampling_test',['cuda']),('local_anchor_migration_test',['cuda']),('variable_anchor_test',['cuda'])]:
        run('lifecycle_'+target,[S/'bin'/target,*args])
    for name,bits in [('legacy_a',0),('adam',1),('adam_decode',3),('fused_a',7),('legacy_b',0),('fused_b',7)]:
        run(name,[S/'bin/optimized_native_benchmark',CP,DATA,R/name,'61',str(bits)])
    # 本gate只使用预注册误差、GradNone和exact identity；不使用质量来挑参数。
    run('fixed_replay_gate',['/home/prmi/anaconda3/envs/zwj/bin/python',R/'verify_replays.py'],env={k:v for k,v in ENV.items() if k!='LD_LIBRARY_PATH'})
    run('scaling',[S/'bin/fused_mapping_scaling_benchmark',CP,Path(F)/'lod_0.050000_parity.pt',R/'scaling.csv'])
    for name,bits in [('profile_legacy',0),('profile_fused',7)]:
        run(name,['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',R/name,
            S/'bin/optimized_native_benchmark',CP,DATA,R/name,'4',str(bits)])
        run(name+'_export',['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','--force-overwrite=true','-o',R/(name+'.sqlite'),R/(name+'.nsys-rep')])
    # 同binary、同默认配置、相同完整轨迹；开关只改变计算实现。
    for name,enabled in [('online_legacy_a',0),('online_fused_a',1),('online_fused_b',1),('online_legacy_b',0)]:
        d=R/name;d.mkdir(exist_ok=False)
        text=(S/'cfg/selector/selector_config.yaml').read_text()
        for key in ['fused_dense_adam','fused_compensation_decode','fused_anchor_input_pack']:
            text=text.replace('CUDA.'+key+': 0','CUDA.'+key+': '+str(enabled))
        config=d/'input_selector.yaml';config.write_text(text)
        env=dict(ENV,CARTGS_MAPPING_TRACE=str(d/'mapping_trace.csv'))
        run(name,[S/'bin/tum_rgbd',S/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',S/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
            S/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',DATA,S/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',d,
            'no_viewer','--selector-config',config],env)
    (R/'validation_complete.json').write_text(json.dumps(dict(passed=True)))
