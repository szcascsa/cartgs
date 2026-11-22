"""固定checkpoint，成对交替A～E；Profiler与正式计时分离。"""
from pathlib import Path
import subprocess, os, json, threading, time
BUILD_ROOT=Path('/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023');R=Path(os.environ.get('CARTGS_PHASE2_RESULTS',str(BUILD_ROOT)));R.mkdir(parents=True,exist_ok=True);BIN=BUILD_ROOT/'source/bin'
CP='/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply'
DATA='/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz'
FIX='/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1/Full_parity.pt'
env=os.environ.copy();env['CARTGS_CUDA_PHASE2_TRACE']='1'
runs=[]
def run(args,name,cwd=R):
    print('RUN',name,flush=True)
    start=time.monotonic();command=[str(x) for x in args]
    with (R/(name+'.log')).open('w') as f:result=subprocess.run(command,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    runs.append(dict(name=name,command=command,cwd=str(cwd),returncode=result.returncode,wall_s=time.monotonic()-start));(R/'suite_commands.json').write_text(json.dumps(runs,indent=2))
    result.check_returncode()
stop=threading.Event()
def monitor():
    with (R/'gpu_load.csv').open('w') as f:
        f.write('time,gpu_util_pct,memory_used_mib,power_w\n')
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
            f.write(str(time.time())+','+p.stdout.strip()+'\n');f.flush();stop.wait(1)
t=threading.Thread(target=monitor);t.start()
try:
    run([BIN/'fused_mapping_adam_test',Path(FIX).parent,R/'production_fixed_adam'],'production_fixed_adam')
    for binary,folder in [('fused_anchor_stats_test','production_phase4'),('fused_selector_sampling_test','production_phase5'),('renderer_workspace_test','production_phase6'),('ssim_backward_test','production_ssim')]:
        run([BIN/binary,CP,FIX,R/folder],folder)
    run([BIN/'phase2_anchor_stats_scaling',R/'anchor_scaling.csv'],'anchor_scaling')
    run([BIN/'phase2_topology_checkpoint_benchmark',CP,R/'topology_checkpoint'],'topology_checkpoint')
    for mode in ['legacy','fused']:
        d=R/('lifecycle_'+mode);d.mkdir(exist_ok=True)
        run([BIN/'private_compensation_mapping_test','cuda',mode],'lifecycle_'+mode,d)
    variants=[('A',0),('B',7),('C',15),('D',31),('E_workspace',63),('E',127)]
    for scale in [1.,.5]:
        for repeat in range(2):
            for name,bits in variants if repeat==0 else variants[::-1]:
                label=f'bench_{scale}_{repeat}_{name}'
                run([BIN/'phase2_staged_mapping_benchmark',CP,DATA,R/label,60,bits,scale],label)
    # Independent-repeat envelope and stateful, fixed-topology Full/LoD sequence.
    for name,bits in [('B_a',7),('B_b',7),('C',15),('D',31),('E_a',127),('E_b',127)]:
        label='continuous_'+name;run([BIN/'phase2_continuous_mapping_replay',CP,DATA,R/label,50,bits],label)
    for name,bits in [('B_a',7),('E_a',127),('E_b',127),('B_b',7)]:
        label='continuous_online_'+name;run([BIN/'phase2_continuous_mapping_replay',CP,DATA,R/label,30,bits,.5],label)
    # Small profiler replay; this timing is never mixed with the formal samples.
    for name,bits in variants:
        label='profile_'+name
        run(['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',R/label,
            BIN/'phase2_staged_mapping_benchmark',CP,DATA,R/(label+'_data'),3,bits,1.],label)
        run(['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','--force-overwrite=true','--output',R/(label+'.sqlite'),R/(label+'.nsys-rep')],label+'_export')
    run(['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o',R/'profile_topology',BIN/'phase2_topology_checkpoint_benchmark',CP,R/'profile_topology_data'],'profile_topology')
    run(['/usr/local/cuda-11.8/bin/nsys','export','--type=sqlite','--force-overwrite=true','--output',R/'profile_topology.sqlite',R/'profile_topology.nsys-rep'],'profile_topology_export')
    # Actual asynchronous SLAM is a smoke/latency measurement. Different final
    # step/topology counts are reported, never used as a kernel-quality gate.
    source=BUILD_ROOT/'source'
    for variant in ['B','E']:
        folder=R/('online_'+variant);folder.mkdir(exist_ok=False);text=(source/'cfg/selector/selector_config.yaml').read_text()
        keys=['fused_dense_adam','fused_compensation_decode','fused_anchor_input_pack']
        if variant=='E':keys+=['fused_anchor_full_stats','fused_selector_compaction','reuse_renderer_workspace','optimized_ssim_backward']
        for key in keys:text=text.replace('CUDA.'+key+': 0','CUDA.'+key+': 1')
        config=folder/'input_selector.yaml';config.write_text(text)
        original_env=env;env=dict(env,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',CARTGS_AUDIT_SEED='2026',CARTGS_MAPPING_TRACE=str(folder/'mapping_trace.csv'),LD_LIBRARY_PATH=str(source/'lib')+':'+env.get('LD_LIBRARY_PATH',''))
        run([BIN/'tum_rgbd',source/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',source/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',source/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',DATA,source/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',folder,'no_viewer','--selector-config',config],'online_'+variant,source)
        env=original_env
    print('SUITE_PASS',flush=True)
finally:
    stop.set();t.join()
