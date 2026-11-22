"""在同一冻结Master上执行22项严格K裁剪，逐字执行CaRtGS原评价循环。"""
from pathlib import Path
from collections import OrderedDict
import argparse
import ast
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import time

from common import ROOT, RATIOS, METHODS, load, write, sha, budget, order_points


class Images:
    def __init__(self,cv2):
        self.cv2=cv2
        self.cache=OrderedDict()
        self.bytes=0
        self.limit=6*1024**3

    def __getattr__(self,name):return getattr(self.cv2,name)

    def imread(self,path,flags):
        key=(path,flags)
        if key in self.cache:return self.cache[key]
        value=self.cv2.imread(path,flags)
        assert value is not None,path
        while self.cache and self.bytes+value.nbytes>self.limit:
            _,old=self.cache.popitem(last=False)
            self.bytes-=old.nbytes
        if value.nbytes<=self.limit:
            self.cache[key]=value
            self.bytes+=value.nbytes
        return value


def setup(group,folder):
    import torch
    import numpy as np
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend='Agg'
    torch.cuda.set_per_process_memory_fraction(0.33)
    snapshot=ROOT/'runtime/python'
    sys.path[:0]=[str(snapshot),str(snapshot/'diff-gaussian-rasterization')]
    script=snapshot/'run.py'
    model=Path(group['vanilla']['checkpoint'])
    sys.argv=[str(script),str(model.parent),group['data'],'--shutdown-model',model.name,
              '--output_path',str(folder),'--no-save-images']
    if group['correct_scale']:sys.argv.append('--correct_scale')
    text=script.read_text()
    marker='    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []'
    assert text.count(marker)==1
    prefix,tail=text.split(marker)
    namespace={'__name__':'__main__','__file__':str(script)}
    exec(compile(prefix,str(script),'exec'),namespace)
    expected=load(Path(group['vanilla']['evaluation'])/'evaluation_frames.json')
    actual=load(folder/'evaluation_frames.json')
    assert expected['gt_indices']==actual['gt_indices']
    assert all(Path(a).samefile(b) for a,b in zip(expected['rgb_paths'],actual['rgb_paths']))
    assert len(actual['rgb_paths'])==len(expected['rgb_paths'])==group['vanilla']['frames']
    count=namespace['gaussians'].get_xyz.shape[0]
    assert count==group['vanilla']['gaussians']
    original_code=compile('if True:\n'+marker+tail,str(script),'exec')
    write(folder/'SETUP.json',dict(source=sha(script),N=count,frames=len(actual['rgb_paths']),
            image_protocol='original_rgb',associations_equal=True,
            correct_scale=group['correct_scale'],cuda_extension_snapshot=True,
            torch_version=torch.__version__,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32))
    return namespace,original_code


def check_full(group,ns,folder):
    import torch
    import numpy as np
    import cv2
    from utils import MiniCam
    from renderer import prepare_render_data,render_prepared
    reference=Path(group['vanilla']['evaluation'])
    history={name:np.loadtxt(reference/(name+'.txt')).reshape(-1) for name in ['psnr','ssim','lpips']}
    count=len(ns['associations'])
    rows=[]
    full=ns['render_data']
    all_rows=prepare_render_data(ns['gaussians'],torch.ones(group['vanilla']['gaussians'],device='cuda',dtype=torch.bool))
    assert all(torch.equal(v,all_rows['inputs'][k]) for k,v in full['inputs'].items() if isinstance(v,torch.Tensor))
    with torch.no_grad():
        for index in [0,count//2,count-1]:
            trajectory,rgb=ns['associations'][index]
            cam=MiniCam(ns['width'],ns['height'],ns['fovx'],ns['fovy'],np.linalg.inv(ns['poses'][trajectory]))
            pred=render_prepared(cam,full,ns['background'])['render']
            assert torch.equal(pred,render_prepared(cam,all_rows,ns['background'])['render'])
            truth=cv2.cvtColor(cv2.imread(ns['gt_color_paths'][rgb],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            truth=torch.from_numpy(np.array(truth)).to('cuda').permute(2,0,1)[None]/255.0
            pred=pred.permute(1,2,0).clamp(0,1).permute(2,0,1)[None]
            row=dict(frame=index,all_row_mask_image_bitwise_equal=True)
            for name in history:
                metric=ns['calc_'+name]
                metric.reset()
                value=metric(pred,truth).item()
                error=abs(value-history[name][index])
                assert error<2e-5,(group['id'],name,index,value,history[name][index])
                row[name+'_abs_error']=error
            rows.append(row)
    write(folder/'FULL_PARITY.json',dict(passed=True,checks=rows))
    return rows


def run(group,folder,limit):
    import torch
    import numpy as np
    from renderer import prepare_render_data
    ns,tail=setup(group,folder/'setup')
    parity=check_full(group,ns,folder)
    opacity=ns['gaussians'].get_opacity.detach().flatten().cpu().numpy()
    count=len(opacity)
    orders={method:order_points(method,opacity) for method in METHODS}
    ns['cv2']=Images(ns['cv2'])
    original_associations=ns['associations']
    if limit:ns['associations']=original_associations[:limit]
    done=[]
    for method in METHODS:
        np.save(folder/(method+'_ranking.npy'),orders[method])
        for ratio in RATIOS:
            output=folder/'evaluation'/method/ratio
            if (output/'metric_summary.json').exists():
                summary=load(output/'metric_summary.json')
                assert summary['ratio']==float(ratio) and summary['actual_K']==budget(ratio,count)
                done.append(summary)
                continue
            if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('DISK_BELOW_5GIB')
            output.mkdir(parents=True,exist_ok=True)
            k=budget(ratio,count)
            selected=np.sort(orders[method][:k])
            assert len(selected)==len(np.unique(selected))==k
            if method=='opacity' and 0<k<count:
                assert opacity[selected].min()>=opacity[orders[method][k:]].max()
            mask=torch.zeros(count,device='cuda',dtype=torch.bool)
            mask[torch.from_numpy(selected).to('cuda')]=True
            ns['render_data']=prepare_render_data(ns['gaussians'],mask)
            assert ns['render_data']['inputs']['means3D'].shape[0]==k
            ns['output_result_path']=str(output)
            ns['selector_stats']=dict(total_gaussians=count,active_gaussians=k,render_gaussians=k,
                selection_policy=method,ratio=float(ratio),compensation_applied=False)
            np.save(output/'selected_ids.npy',selected)
            write(output/'selection.json',dict(method=method,ratio=float(ratio),N=count,K=k,seed=0,
                nested=True,opacity='activated sigmoid',original_row_order=True,
                selected_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                opacity_mean=float(opacity[selected].mean()) if k else None,
                checkpoint=group['vanilla']['checkpoint']))
            for name in ['psnr','ssim','lpips']:ns['calc_'+name].reset()
            started=time.monotonic()

            def progress(length,**unused):
                for index in range(length):
                    yield index
                    if (index+1)%32==0 or index+1==length:
                        progress_value=dict(group=group['id'],method=method,ratio=ratio,
                              frame=index+1,frames=length,completed_in_process=len(done),
                              elapsed_seconds=time.monotonic()-started,updated_unix=time.time())
                        write(folder/'PROGRESS.json',progress_value)
                        write(ROOT/f'PROGRESS_{os.getpid()}.json',progress_value)
            ns['trange']=progress
            with torch.no_grad():exec(tail,ns)
            rows=[]
            for index,(trajectory,rgb) in enumerate(ns['associations']):
                row=dict(frame=index,trajectory_index=int(trajectory),rgb_index=int(rgb),
                         rgb_timestamp=float(ns['gt_tstamp'][rgb]),N=count,K=k)
                for name in ['psnr','ssim','lpips']:
                    row[name]=float(ns[name+'_list'][index])
                    assert math.isfinite(row[name])
                row['render_ms']=float(ns['time_list'][index])
                rows.append(row)
            temp=output/'per_frame_metrics.csv.tmp'
            with temp.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
            temp.replace(output/'per_frame_metrics.csv')
            summary=dict(group=group['id'],method=method,ratio=float(ratio),seed=0,N=count,
                target_K=k,actual_K=k,actual_ratio=k/count,frames=len(rows),images_saved=0,
                compensation_applied=False,image_protocol='original_rgb',
                correct_scale=group['correct_scale'],seconds=time.monotonic()-started,
                checkpoint=group['vanilla']['checkpoint'],output=str(output))
            for name in ['psnr','ssim','lpips']:
                summary[name]=float(np.mean(ns[name+'_list']))
                assert abs(sum(row[name] for row in rows)/len(rows)-summary[name])<1e-9
            summary['render_ms']=float(np.mean(ns['time_list']))
            summary['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
            write(output/'metric_summary.json',summary)
            done.append(summary)
            print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)
    assert len(done)==22
    write(folder/'COMPLETE.json',dict(passed=True,group=group['id'],evaluations=done,
          full_parity=parity,full_frames=len(original_associations),smoke=bool(limit)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',required=True)
    parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    snapshot=ROOT/'runtime/python'
    sys.path[:0]=[str(snapshot),str(snapshot/'diff-gaussian-rasterization')]
    group=next(g for g in load(ROOT/'EXPERIMENTS.json')['groups'] if g['id']==args.group)
    folder=ROOT/('smoke' if args.limit else 'runs')/group['id'];folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (folder/'COMPLETE.json').exists():return
    run(group,folder,args.limit)


if __name__=='__main__':main()
