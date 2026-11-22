from pathlib import Path
import hashlib
import json
import subprocess
import torch
D=Path(__file__).resolve().parent
ROOT=Path('/home/prmi/zwj/mycartgs')
results={}
def compare(a,b):
    x,y=[torch.jit.load(str(p),map_location='cpu').state_dict() for p in [a,b]]
    assert x.keys()==y.keys()
    rows=[]
    for name,t in x.items():
        delta=(t-y[name]).abs()
        rows.append({'tensor':name,'max_abs':float(delta.max()) if delta.numel() else 0,
                     'mean_abs':float(delta.float().mean()) if delta.numel() else 0,
                     'bitwise_equal':torch.equal(t,y[name]),'allclose':torch.allclose(t,y[name],atol=2e-5,rtol=2e-4)})
    rec={'allclose':all(r['allclose'] for r in rows),'bitwise_equal':all(r['bitwise_equal'] for r in rows),'max_abs':max(r['max_abs'] for r in rows),'rows':rows}
    assert rec['allclose'],(a,b)
    return rec
for branch in ['tf','selector','both']:
    for direction in ['forward','reverse']:
        rel='raw/paired_production_'+direction+'_'+branch+'/warm_0/final_parameters.pt'
        results[direction+'_'+branch]=compare(D/'baseline_production'/rel,D/'production'/rel)
    for stage in ['baseline_production','production']:
        results[stage+'_repeat_'+branch]=compare(D/stage/'raw'/('paired_production_forward_'+branch)/'warm_0/final_parameters.pt',D/stage/'raw'/('paired_production_reverse_'+branch)/'warm_0/final_parameters.pt')
(D/'production_parity.json').write_text(json.dumps(results,indent=2))
hashes={}
for name in ['lib/libgaussian_mapper.so','lib/libgaussian_viewer.so','bin/tum_rgbd','build-cuda118-sm86-gcc9/libtransform_field_cuda.a']:
    p=ROOT/name;hashes[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for p in (D/'formal_libraries').glob('*.a'):
    hashes['baseline_'+p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(D/'deployment/binary_manifest.json').write_text(json.dumps(hashes,indent=2))
with (D/'deployment/nvidia-smi.txt').open('w') as f:subprocess.run(['nvidia-smi','-q'],stdout=f,check=True)
for stage,library in [('baseline_production',D/'deployment/backup/lib'),('production',ROOT/'lib')]:
    import os
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(library)+':'+env.get('LD_LIBRARY_PATH','')
    (D/stage/'ldd.txt').write_text(subprocess.check_output(['ldd',str(D/stage/'benchmark_native')],text=True,env=env))
print({k:{x:y for x,y in v.items() if x!='rows'} for k,v in results.items()})
