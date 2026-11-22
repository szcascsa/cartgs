"""验证实测占空比上限、全SM覆盖和v21准备边界。"""
import argparse
import csv
import fcntl
import time

from common import *
from phase_gate import RenderLoad, validate_phase_intervals
from renderer_reference import Renderer


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate_duties(path, hardware):
    pulses = read(path)
    phases = read(Path(str(path)+'.phases.csv'))
    assert phases
    assert all(int(row['unique_sms'])==hardware['sm_count'] for row in pulses)
    assert hardware['resident_threads_per_sm']==hardware['maximum_threads_per_sm']
    assert all(int(row['blocks_each_sm'])==hardware['resident_blocks_per_sm'] for row in pulses)
    for row in phases:
        target = float(row['target_duty'])
        assert 0 <= target <= .8
        assert float(row['measured_resident_duty']) <= target+1e-8, row
        assert float(row['charged_duty']) <= target+1e-8, row
        assert (int(row['pulses']) > 0) == (target > 0)
    return phases


def target_match(phases):
    active = [row for row in phases if float(row['target_duty']) > 0]
    return dict(tolerance=.05, cases=len(active),
        passed=all(abs(float(row['measured_resident_duty'])-float(row['target_duty'])) <= .05
                   for row in active),
        maximum_absolute_error=max(abs(float(row['measured_resident_duty'])-float(row['target_duty']))
                                   for row in active))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    lock=(ROOT/'reports/run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=ROOT/'runs'/args.label
    folder.mkdir()
    protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
    assert all(sha(name)==value for name,value in protected.items())
    status=dict(state='validating_load',label=args.label,process=identity())
    save(ROOT/'STATUS.json',status)
    save(folder/'ENVIRONMENT.json',dict(gpu=gpu_state('before'),torch=torch.__version__,
        cuda=torch.version.cuda,binary_sha256=sha(V21_BINARY),phase_seconds=.15,
        target_match_observation_tolerance=.05))
    hardware=None
    try:
        path=folder/'WINDOW_PULSES.csv'
        load=RenderLoad(ROOT/'build/librender_load.so',path)
        hardware=load.information
        save(folder/'LOAD_HARDWARE.json',hardware)
        for invalid in (-1., .80001, 1., float('nan'), float('inf')):
            assert load.library.render_load_resume(load.pointer,-1,invalid)==-1
        try:
            for repeat in range(3):
                for stage,duty in enumerate((0.,.2,.4,.6,.8,0.)):
                    load.resume(stage,duty)
                    time.sleep(.15)
                    load.pause()
        finally:
            load.close()
        windows=validate_duties(path,hardware)
        matching=target_match(windows)
        save(folder/'LONG_WINDOW_TARGET_MATCH.json',matching)
        assert matching['passed'], matching
        path=folder/'DESTRUCTOR_PULSES.csv'
        load=RenderLoad(ROOT/'build/librender_load.so',path)
        load.resume(0,.8)
        time.sleep(.01)
        load.close()
        destructor=validate_duties(path,hardware)
        with torch.inference_mode(False),torch.no_grad():
            renderer=Renderer()
        path=folder/'V21_PULSES.csv'
        load=RenderLoad(ROOT/'build/librender_load.so',path)
        frames=[]
        expected={}
        try:
            with torch.inference_mode():
                for budget in (1000,4000):
                    for _ in range(5):
                        renderer.call('ours_fixed4000',budget)
                    _,image=renderer.call('ours_fixed4000',budget)
                    expected[budget]=hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
                for stage,duty in enumerate((0.,.2,.4,.6,.8,0.)):
                    for budget in (1000,4000):
                        for _ in range(5):
                            record,image=renderer.call('ours_fixed4000',budget,load,stage,duty)
                            frames.append(record)
                            signature=hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()
                            assert signature==expected[budget],(stage,budget)
        finally:
            load.close()
            renderer.close()
        phases=validate_duties(path,hardware)
        packets=read(path)
        audit=validate_phase_intervals(frames,packets)
        write_csv(folder/'V21_TIMINGS.csv',frames)
        assert all(sha(name)==value for name,value in protected.items())
        result=dict(passed=True,hardware=hardware,long_window_cases=len(windows),v21_requests=len(frames),
            v21_phases=len(phases),v21_images_bitwise=True,maximum_requested_duty=.8,
            long_window_target_match=matching,short_window_target_match=target_match(phases),
            destructor_off_settlement=True,
            maximum_measured_duty=max(float(row['measured_resident_duty']) for row in windows+phases+destructor),
            maximum_charged_duty=max(float(row['charged_duty']) for row in windows+phases+destructor),
            pulse_count=len(packets),prepare_overlap=audit['artificial_load_prepare_overlap'],
            protected_files=len(protected),saved_render_images=0)
        save(folder/'VALIDATION.json',result)
        save(folder/'GPU_AFTER.json',gpu_state('after'))
        status.update(state='complete',result=result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        status.update(state='failed',error=repr(error))
        raise
    finally:
        save(ROOT/'STATUS.json',status)


if __name__=='__main__':
    main()
