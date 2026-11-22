"""Gate E: complete TUM online runs and full-frame Full/LoD evaluation.

Independent SLAM runs are not fixed-schedule causal experiments. Only files
created below this run's output root may be removed after metrics complete.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/newgi/ultralight_mode2_20261014'
DATA = Path('/home/prmi/datasets/TUM_RGBD')
SCENES = {
    'room': 'freiburg1_room', 'desk': 'freiburg1_desk',
    'xyz': 'freiburg2_xyz', 'long': 'freiburg3_long_office_household',
}
MODES = ['original_full_only','mode1','mode2']
RATIOS = [0.01,0.05,0.10,0.15]


def override_yaml(source, target, values):
    pending = dict(values)
    lines = []
    for line in source.read_text().splitlines():
        key = line.split(':',1)[0]
        if key in pending:
            lines.append(key + ': ' + str(pending.pop(key)))
        else:
            lines.append(line)
    lines.extend(k + ': ' + str(v) for k,v in pending.items())
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(lines)+'\n')


def environment():
    env = os.environ.copy()
    # evo's saved backend overrides MPLBACKEND. Isolate it without modifying
    # the user's ~/.evo settings or forcing pretrained-model downloads.
    home = OUT / 'evaluation_home'
    settings = home / '.evo/settings.json'
    settings.parent.mkdir(parents=True,exist_ok=True)
    original = Path.home() / '.evo/settings.json'
    current = json.loads(original.read_text()) if original.exists() else {}
    current['plot_backend'] = 'Agg'
    settings.write_text(json.dumps(current,indent=2)+'\n')
    env.update(HOME=str(home),MPLBACKEND='Agg',TORCH_HOME='/home/prmi/.cache/torch',
               PYTHONUNBUFFERED='1')
    return env


def execute(command, log, env, timeout):
    log.parent.mkdir(parents=True,exist_ok=True)
    begin = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run([str(x) for x in command],cwd=ROOT,env=env,
                                stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    record = {'command':[str(x) for x in command],'exit':result.returncode,
              'seconds':time.monotonic()-begin,'log':str(log)}
    print(json.dumps(record),flush=True)
    if result.returncode:
        raise RuntimeError(f'Run failed: {log}')
    return record


def ply_count(checkpoint):
    files = list((checkpoint/'ply/point_cloud').glob('iteration_*/point_cloud.ply'))
    assert len(files) == 1, files
    with files[0].open('rb') as stream:
        for _ in range(256):
            line = stream.readline()
            if line.startswith(b'element vertex '):
                return int(line.split()[-1])
    raise RuntimeError('PLY vertex count missing')


def metrics(folder):
    result = {}
    for line in (folder/'eval.txt').read_text().splitlines():
        if ':' in line:
            key,value = line.split(':',1)
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                pass
    # evo prints whitespace-delimited statistics, not key:value records.
    # Keep translation and rotation separate rather than overwriting rmse.
    trajectory_part = None
    for line in (folder/'metrics_traj.txt').read_text().splitlines():
        if line.startswith('APE w.r.t. translation'):
            trajectory_part = 'translation'
        elif line.startswith('APE w.r.t. rotation'):
            trajectory_part = 'rotation'
        elif trajectory_part:
            fields = line.split()
            if len(fields) == 2 and fields[0] in {'max','mean','median','min','rmse','sse','std'}:
                unit = '_m' if trajectory_part == 'translation' and fields[0] != 'sse' else ''
                result[f'ape_{trajectory_part}_{fields[0]}{unit}'] = float(fields[1])
    stats = folder/'selector_stats.json'
    if stats.exists():
        result['selector_stats'] = json.loads(stats.read_text())
    # Actual evaluated-frame count, not the number of selected keyframes.
    result['evaluated_frames'] = len((folder/'psnr.txt').read_text().splitlines())
    return result


def cleanup_images(run):
    removed = 0
    root = run.resolve()
    for file in run.rglob('*'):
        if file.suffix.lower() not in {'.png','.jpg','.jpeg'} or file.is_symlink():
            continue
        if file.is_file() and file.resolve().is_relative_to(root):
            file.unlink()
            removed += 1
    return removed


def evaluation_view(run, checkpoint, phase):
    # The unchanged evaluator accepts *_shutdown checkpoints. A read-only
    # symlink view exposes the online snapshot under that existing convention.
    view = run / ('eval_source_'+phase)
    view.mkdir(exist_ok=True)
    for file in run.iterdir():
        if file.is_file() and not (view/file.name).exists():
            (view/file.name).symlink_to(file.resolve())
    name = checkpoint.name.split('_',1)[0] + '_shutdown'
    link = view/name
    if not link.exists():
        link.symlink_to(checkpoint.resolve(),target_is_directory=True)
    return view,link


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes',nargs='+',choices=SCENES,default=list(SCENES))
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--refresh-metrics-only',action='store_true')
    args = parser.parse_args()
    if args.refresh_metrics_only:
        # Run only after the online driver has exited; no training or rendering.
        summary = OUT/'summary.json'
        report = json.loads(summary.read_text())
        for record in report['runs']:
            assert record['complete'], 'Do not refresh an active/incomplete batch'
            for phase,quality in record['quality'].items():
                folder = Path(record['output'])/'metrics'/phase
                if record['mode'] == 'mode1' and phase == 'final':
                    folder = Path(record['output'])/'metrics/online'
                quality['full'] = metrics(folder/'full')
                for ratio in quality.get('lod',{}):
                    quality['lod'][ratio] = metrics(folder/f'ratio_{float(ratio):.2f}')
        summary.write_text(json.dumps(report,indent=2)+'\n')
        return
    accepted = json.loads((HERE/'acceptance.json').read_text())
    assert accepted['complete'], 'A/B/C/D gates must all pass before E'
    for file,digest in accepted['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, f'Source changed after gates: {file}'
    OUT.mkdir(parents=True,exist_ok=True)
    env = environment()
    # Prove real run.py imports with its CUDA metrics and headless evo backend.
    execute([sys.executable,'-c',"import sys;sys.path.insert(0,'python');import run;print('EVAL_IMPORT_OK')"],
            OUT/'evaluation_preflight.log',env,180)
    if args.preflight_only:
        return
    report = {'acceptance':str(HERE/'acceptance.json'),'runs':[],
              'protocol':'complete input sequences, independent online SLAM, all RGB evaluation frames; no causal CUDA quality claim'}
    summary = OUT/'summary.json'
    if summary.exists():
        report = json.loads(summary.read_text())
    completed = {(r['scene'],r['mode']) for r in report['runs'] if r.get('complete')}
    for scene in args.scenes:
        name = SCENES[scene]
        dataset = DATA/('rgbd_dataset_'+name)
        association = ROOT/'cfg/ORB_SLAM3/RGB-D/TUM/associations'/('tum_'+name+'.txt')
        for mode in args.modes:
            if (scene,mode) in completed:
                continue
            if shutil.disk_usage(OUT).free < 5*1024**3:
                raise RuntimeError('Less than 5 GiB free; do not remove existing user results')
            run = OUT/scene/mode
            if run.exists():
                raise RuntimeError(f'Incomplete run already exists, inspect before retry: {run}')
            run.mkdir(parents=True)
            values = {'Training.ultra_light_mode':0 if mode=='original_full_only' else 1 if mode=='mode1' else 2,
                      'Training.force_tail_optimization_enabled':0,'Training.tail_optimization_steps':-1,
                      'Selector.max_mapping_iterations':0,'GradientConstraint.enable':0,
                      'Selector.replay_enabled':0,'DriftStability.enable':0,
                      'Selector.lod_render_scale':0.5,'TransformField.backend':'"cuda"'}
            if mode == 'original_full_only':
                values.update({'Selector.enabled':0,'TransformField.enable':0,
                               'GlobalGI.mode':'"disabled"','RecentWindow.enable':0,'ChangeDistill.enable':0})
            config = run/'input_selector.yaml'
            override_yaml(HERE/'input_fast.yaml',config,values)
            command = [ROOT/'bin/tum_rgbd',ROOT/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
                       ROOT/'cfg/ORB_SLAM3/RGB-D/TUM'/('tum_'+name+'.yaml'),
                       ROOT/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',dataset,association,run,
                       'no_viewer','--selector-config',config]
            record = {'scene':scene,'mode':mode,'complete':False,'output':str(run)}
            report['runs'].append(record)
            summary.write_text(json.dumps(report,indent=2)+'\n')
            record['training'] = execute(command,run/'training.log',env,2400)
            log_text = (run/'training.log').read_text(errors='replace')
            record['mapping_performance_logs'] = [line for line in log_text.splitlines()
                                                   if line.startswith(('[Mapping PERF FINAL]','[Mapping PERF PHASE]'))]
            final = max(run.glob('*_shutdown'),key=lambda p:int(p.name.split('_',1)[0]))
            online = list(run.glob('*_online_end'))
            checkpoints = {'final':final,'online':online[0] if online else final}
            if mode == 'mode1':
                checkpoints = {'online':final} # Mode1 has no tail, hence no second map.
            record['quality'] = {}
            for phase,checkpoint in checkpoints.items():
                view,link = evaluation_view(run,checkpoint,phase)
                folder = run/'metrics'/phase
                record['quality'][phase] = {'gaussian_total':ply_count(checkpoint)}
                state = checkpoint/'ply'/('ultralight_mode2_state.json' if mode=='mode2' else 'mapping_stage_state.json')
                record['quality'][phase]['stage_counts'] = json.loads(state.read_text())
                full = folder/'full'
                execute([sys.executable,ROOT/'python/run.py',view,dataset,'--output_path',full],
                        folder/'full.log',env,2400)
                record['quality'][phase]['full'] = metrics(full)
                if mode != 'original_full_only':
                    record['quality'][phase]['lod'] = {}
                    for ratio in RATIOS:
                        target = folder/f'ratio_{ratio:.2f}'
                        execute([sys.executable,ROOT/'python/run.py',view,dataset,'--output_path',target,
                                 '--selector',link/'ply/selector.pt','--selector-ratio',str(ratio),
                                 '--selector-config',link/'ply/selector_config.yaml'],folder/f'ratio_{ratio:.2f}.log',env,2400)
                        record['quality'][phase]['lod'][str(ratio)] = metrics(target)
                record['removed_generated_images'] = record.get('removed_generated_images',0)+cleanup_images(run)
                summary.write_text(json.dumps(report,indent=2)+'\n')
            if mode == 'mode1':
                record['quality']['final'] = json.loads(json.dumps(record['quality']['online']))
            record['complete'] = True
            summary.write_text(json.dumps(report,indent=2)+'\n')
            print(f'COMPLETE {scene}/{mode}',flush=True)


if __name__ == '__main__':
    main()
