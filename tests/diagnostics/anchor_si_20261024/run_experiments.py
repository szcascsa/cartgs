"""预注册的两 seed 冻结地图和真实在线实验；不按评价质量挑选终点。"""
from pathlib import Path
import csv,json,os,re,subprocess,threading,time
root=Path('/home/prmi/zwj/anchor-si-20261024');source=root/'source';binary=source/'bin'
checkpoint='/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply'
data=Path('/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz')
env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',LD_LIBRARY_PATH=str(source/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
commands=json.loads((root/'commands.json').read_text()) if (root/'commands.json').exists() else []
completed={row['label'] for row in commands if row['returncode']==0 and row['label']!='validation'}
stop=threading.Event()
def monitor():
    with (root/'gpu_load.csv').open('w') as log:
        log.write('unix_time,gpu_util,memory_used_mib\n')
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            log.write(str(time.time())+','+p.stdout.strip()+'\n');log.flush();stop.wait(2)
def run(args,label,extra_env=None):
    if label in completed:
        print('PRESERVE_COMPLETED',label,flush=True);return
    print('RUN',label,flush=True);start=time.time();args=list(map(str,args))
    with (root/(label+'.log')).open('w') as log:
        result=subprocess.run(args,cwd=source,env=dict(env,**(extra_env or {})),stdout=log,stderr=subprocess.STDOUT)
    commands.append(dict(label=label,args=args,returncode=result.returncode,wall_s=time.time()-start))
    (root/'commands.json').write_text(json.dumps(commands,indent=2));result.check_returncode()
thread=threading.Thread(target=monitor);thread.start()
try:
    run([binary/'anchor_si_test',checkpoint,root/'validation'],'validation')
    for seed,arms in [(2026,['off','gaussian','anchor']),(2027,['anchor','gaussian','off'])]:
        for arm in arms:
            output=root/f'frozen_{arm}_{seed}';assert not output.exists() or output.name in completed
            run([binary/'anchor_si_frozen_train',checkpoint,data,output,arm,seed,1600],output.name)
    for arm,seed in [('off',2026),('anchor',2026),('anchor',2027),('off',2027)]:
        label=f'online_{arm}_{seed}';output=root/label
        if label in completed:continue
        output.mkdir(exist_ok=False)
        config=(source/'cfg/selector/selector_config.yaml').read_text()
        config=re.sub(r'(?m)^AnchorSI.enabled:.*$',f'AnchorSI.enabled: {int(arm=="anchor")}',config)
        path=output/'input_selector.yaml';path.write_text(config)
        run([binary/'tum_rgbd',source/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
             source/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',source/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
             data,source/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',output,'no_viewer','--selector-config',path],
            label,dict(CARTGS_AUDIT_SEED=str(seed),CARTGS_MAPPING_TRACE=str(output/'mapping_trace.csv')))
    (root/'experiments_complete.json').write_text(json.dumps(dict(passed=True,commands=len(commands))))
finally:stop.set();thread.join()
