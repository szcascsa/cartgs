from pathlib import Path
import subprocess,json,time,sys,os

HERE=Path(__file__).resolve().parent;R=HERE/'reports';p=R/'COMPONENT_CONTROLLER.json'
assert not p.exists()
state=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state='running',accepted=[],rejected=['components_final02','components_final03'])
save=lambda:p.write_text(json.dumps(state,indent=2))
save()
try:
    for group in range(8):
        for attempt in range(1,5):
            name=f'components_g{group}_a{attempt}'
            subprocess.run([sys.executable,'-B','launch.py','--measure',name,sys.executable,'-B','-u','components_preparation.py','--output',name,'--slice',str(group)],cwd=HERE,check=True)
            while True:
                j=json.loads((R/'jobs'/(name+'.json')).read_text())
                if j.get('state') in ['failed','complete']:break
                time.sleep(.3)
            if j['returncode']!=0:raise RuntimeError((name,'No blind retry'))
            if j['measurement_valid']:
                state['accepted'].append(name);save();print('COMPONENT_BATCH_VALID',name,flush=True);break
            state['rejected'].append(name);save()
        else:raise RuntimeError(('Repeated external load',group))
    state['state']='complete';(R/'COMPONENT_MEASUREMENT.json').write_text(json.dumps(state,indent=2))
except Exception as error:state['state']='failed';state['error']=str(error);raise
finally:save()
