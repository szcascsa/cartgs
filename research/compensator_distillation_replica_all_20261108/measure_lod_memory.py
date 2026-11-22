"""同地图Full/动态LoD显存实测：独立进程、原Renderer、不修改模型。"""
from scene_runtime import *
import argparse
import csv
import gc
import subprocess
from deployment_runtime import DeploymentRenderer
from gaussian_model import GaussianModel


def memory():
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())


def buffers(packet):
    tensors=[v for v in packet['inputs'].values() if isinstance(v,torch.Tensor)]
    unique={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in tensors}
    return dict(logical_input_bytes=sum(v.numel()*v.element_size() for v in tensors),
        storage_input_bytes=sum(unique.values()))


@torch.no_grad()
def worker(slug,mode,output,frame_indices=None,student_checkpoint=None):
    scene=next(s for s in SCENES if s['slug']==slug)
    checkpoint=Path(scene['checkpoint'])
    calibration,views=full_trajectory_views(checkpoint,Path(scene['dataset']))
    frames=frame_indices if frame_indices is not None else [19,len(views)//5,2*len(views)//5,3*len(views)//5,4*len(views)//5,len(views)-20]
    c=calibration
    cameras=[MiniCam(c['width'],c['height'],focal2fov(c['fx'],c['width']),focal2fov(c['fy'],c['height']),
        np.linalg.inv(np.asarray(views[frame]['pose']))) for frame in frames]
    background=torch.zeros(3,device='cuda')
    torch.cuda.synchronize();initial=memory()
    engine=None;master=None
    if mode=='full_map_only':
        pc=GaussianModel(3)
        ply=list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'));assert len(ply)==1
        pc.load_ply(str(ply[0]));master=prepare_render_data(pc)
        # 此路径只保留未补偿Master的最终推理属性，不加载Selector/Feature，也不保留训练态副本。
        del pc
    else:
        engine=DeploymentRenderer.from_checkpoint(checkpoint,student_checkpoint=student_checkpoint)
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    loaded=memory();rows=[]
    modes=['runtime_full',.01,.05,.1,.3,.5,.7] if mode=='runtime_switch' else [mode]
    for cycle in range(2):
        for request in modes:
            for frame,cam in zip(frames,cameras):
                gc.collect();torch.cuda.synchronize()
                before=memory();torch.cuda.reset_peak_memory_stats()
                if request=='full_map_only':packet=master;ids=None;stats={}
                elif request=='runtime_full':packet,ids=engine.full();stats={}
                else:packet,ids,stats=engine.prepare(cam,ratio_budget=float(request))
                torch.cuda.synchronize();prepared=memory();sizes=buffers(packet)
                torch.cuda.reset_peak_memory_stats()
                result=render_prepared(cam,packet,background,return_radii=True)
                torch.cuda.synchronize();rendered=memory()
                row=dict(scene=scene['id'],map_N=scene['gaussians'],mode=str(request),worker_mode=mode,
                    cycle=cycle,frame=frame,width=c['width'],height=c['height'],
                    K=len(packet['inputs']['means3D']),projected_K=int((result['radii']>0).sum()),
                    baseline_allocated=before['allocated'],baseline_reserved=before['reserved'],
                    prepared_allocated=prepared['allocated'],prepare_peak_allocated=prepared['peak_allocated'],
                    prepare_peak_delta=prepared['peak_allocated']-before['allocated'],
                    render_peak_allocated=rendered['peak_allocated'],render_peak_reserved=rendered['peak_reserved'],
                    render_peak_delta=rendered['peak_allocated']-prepared['allocated'],
                    render_output_bytes=sum(v.numel()*v.element_size() for v in [result['render'],result['radii']]),
                    whole_step_peak_allocated=max(prepared['peak_allocated'],rendered['peak_allocated']),
                    graph_count=stats.get('graph_count',0),**sizes)
                row['selection_sha256']=None if ids is None else hashlib.sha256(ids.cpu().contiguous().numpy().tobytes()).hexdigest()
                del result,packet,ids
                gc.collect();torch.cuda.synchronize();row['retained_after']=torch.cuda.memory_allocated()
                rows.append(row)
    gc.collect();torch.cuda.synchronize()
    result=dict(scene=scene['id'],mode=mode,initial=initial,loaded=loaded,final=memory(),
        resident_after_load=loaded['allocated']-initial['allocated'],rows=rows,
        cuda_allocator_backend=torch.cuda.get_allocator_backend(),saved_images=0,
        timing_claimed=False,scope='PyTorch CUDA allocator only; nvidia context/driver/library external allocations excluded',
        student_checkpoint=None if student_checkpoint is None else str(student_checkpoint),
        student_sha256=None if student_checkpoint is None else sha(student_checkpoint),
        full_only_scope='inference packet only; static Master attributes retained; no network/Feature training state')
    save(output,result)
    print('MEMORY_CASE_DONE',slug,mode,result['resident_after_load'],max(r['render_peak_delta'] for r in rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slug');parser.add_argument('--mode')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.slug:
        worker(args.slug,args.mode,args.output);return
    args.output.mkdir(parents=True,exist_ok=False)
    # 各条件新进程，避免前一档CUDAGraph/Allocator峰值影响后一档；另测真实切换累积。
    cases=['full_map_only','runtime_full','0.01','0.05','0.1','0.3','0.5','0.7','runtime_switch']
    scenes=['Replica__office2__rgbd','Replica__room0__rgbd']
    for slug in scenes:
        for mode in cases:
            log=args.output/f'{slug}__{mode}.log'
            with log.open('x') as f:
                cmd=[sys.executable,str(Path(__file__).resolve()),'--slug',slug,'--mode',mode,
                    '--output',str(args.output/f'{slug}__{mode}.json')]
                p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            assert p.returncode==0,(slug,mode,log.read_text()[-2000:])
            print('MEMORY_CASE_DONE',slug,mode,flush=True)
    save(args.output/'COMPLETE.json',dict(passed=True,scenes=scenes,cases=cases,isolated_workers=18,
        shared_gpu=True,speed_results_valid=False,dram_hardware_counters_measured=False,
        reason='ERR_NVGPUCTRPERM, RmProfilingAdminOnly=1; other training active',models_changed=False))


if __name__=='__main__':main()
