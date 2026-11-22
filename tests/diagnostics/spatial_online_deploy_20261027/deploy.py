"""直接复用现有生产build；先备份再覆盖，仅更新manifest列出的文件。"""
import hashlib,json,os,pathlib,shutil,subprocess,time,zipfile
STAGE=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path('/home/prmi/zwj/mycartgs')
BUILD=ROOT/'build-performance-20261023-121515';state=dict(phase='sync',started=time.time())
def save():(STAGE/'status.json').write_text(json.dumps(state,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,label):
    state['phase']=label;save()
    with (STAGE/(label+'.log')).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
    manifest=json.loads((STAGE/'manifest.json').read_text());changes=[]
    with zipfile.ZipFile(STAGE/'payload.zip') as z:
        assert set(z.namelist())==set(manifest['files'])
        for name,digest in manifest['files'].items():
            target=ROOT/name
            assert target.resolve().is_relative_to(ROOT) and hashlib.sha256(z.read(name)).hexdigest()==digest
            prior=sha(target) if target.is_file() else None
            if prior==digest:continue
            backup=STAGE/'backup/source'/name
            if target.is_file() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
            changes.append(dict(path=name,before=prior,after=digest));target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    for folder,names in [('bin',['tum_rgbd','tum_mono','replica_rgbd','replica_mono','vector_mono','vector_stereo','view_result','train_colmap']),('lib',['libgaussian_mapper.so','libcuda_rasterizer.so'])]:
        for name in names:
            source=ROOT/folder/name;backup=STAGE/'backup/artifacts'/folder/name
            if source.is_file() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,backup)
    (STAGE/'changes.json').write_text(json.dumps(changes,indent=2));state['changed_files']=len(changes);save()
    assert all(sha(ROOT/name)==digest for name,digest in manifest['files'].items())
    cmake='/home/prmi/.local/bin/cmake'
    run([cmake,'-S',str(ROOT),'-B',str(BUILD),'-DCARTGS_BUILD_TESTS=ON'],'configure')
    targets=['spatial_compensation_test','spatial_mapping_test','tum_rgbd','tum_mono','replica_rgbd','replica_mono','vector_mono','vector_stereo','view_result','train_colmap']
    run([cmake,'--build',str(BUILD),'-j2','--target',*targets],'build')
    state.update(phase='built',finished=time.time(),targets=targets,source_files_verified=len(manifest['files']),build=str(BUILD));save();print(json.dumps(state),flush=True)
except Exception as e:state.update(phase='failed',error=str(e));save();raise
