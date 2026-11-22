"""fr1_xyz真实SLAM生产默认验证；只打开非同步trace，不改变训练预算。"""
import argparse,hashlib,json,os,pathlib,subprocess,time
parser=argparse.ArgumentParser();parser.add_argument('--name',default='online_spatial');a=parser.parse_args()
STAGE=pathlib.Path(__file__).resolve().parent;REPO=pathlib.Path('/home/prmi/zwj/mycartgs');OUTPUT=STAGE/a.name
OUTPUT.mkdir(exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',CARTGS_AUDIT_SEED='2026',
    CARTGS_MAPPING_TRACE=str(OUTPUT/'mapping_trace.csv'),
    LD_LIBRARY_PATH=':'.join([str(REPO/'lib'),str(REPO/'third_party/libtorch-cu118/libtorch/lib'),str(REPO/'third_party/install/opencv-cuda118-sm86-gcc9/lib'),'/usr/local/cuda-11.8/lib64',os.environ.get('LD_LIBRARY_PATH','')]))
cmd=[REPO/'bin/tum_rgbd',REPO/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',REPO/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
    REPO/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
    REPO/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',OUTPUT,'no_viewer','--selector-config',REPO/'cfg/selector/selector_config.yaml']
(OUTPUT/'launch.json').write_text(json.dumps(dict(command=list(map(str,cmd)),selector_sha256=hashlib.sha256((REPO/'cfg/selector/selector_config.yaml').read_bytes()).hexdigest(),budget='production natural stop; no forced 10000 or additional cap',profiling='existing asynchronous sampled CUDA events; iteration_logging=0; per-mapping host trace'),indent=2))
start=time.monotonic()
with (OUTPUT/'training.log').open('w') as f:result=subprocess.run(list(map(str,cmd)),cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
record=dict(returncode=result.returncode,wall_seconds=time.monotonic()-start,checkpoint=[str(p) for p in OUTPUT.glob('*_shutdown/ply')])
(OUTPUT/'completion.json').write_text(json.dumps(record,indent=2));print('ONLINE_COMPLETE',json.dumps(record),flush=True)
assert result.returncode==0
