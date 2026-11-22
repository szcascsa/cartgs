"""Student当前位姿与GPU K边界；同一182像素核、严格逐位调度对照。"""
from joint_final_study import *
from joint_release_runtime import JointReleaseRenderer
from tail_strong_runtime import TailStrongRenderer
from tail_cutoff_runtime import TailCutoffRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--sanitizer',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    spec=student_specs(tune=True)[0];source,student,_=load_case(spec)
    factory=NativeCameraPreparer(p.MiniCam);bg=source.scene.background
    mode=json.loads((HERE/'reports/SELECTED_RASTER_20261115.json').read_text())['mode']
    network=network_variants(source,student)['tensor_best'][0]
    renderer=TailStrongRenderer(mode) if mode in [301,302] else TailCutoffRenderer(203) if mode==203 else EncodedSignRenderer(182)
    settings=json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())['config']
    engine=JointReleaseRenderer(source,student,mode,**{k:v for k,v in settings.items() if k!="members"})
    rows=[];retained=[]
    conditions=['fixed0','fixed1','fixed31','fixed32','fixed33','fixed255','fixed256','fixed257','fixed511','fixed512','fixed513',
        'fixed1023','fixed1024','fixed1025','local0.01','local0.1','local0.3','local0.5','full']
    if args.sanitizer:conditions=['fixed0','fixed1','fixed32','fixed33','fixed256','fixed257','fixed512','fixed513','local0.1']
    views=motions(source,quick=True)
    for motion,frame in views:
        cam=factory.from_source(source,frame)
        for condition in conditions:
            packet,ids,_=p.request(network,cam,condition);reference,radii=renderer.render(cam,packet,bg)
            image,actual_radii,actual_ids,_=engine.render_from_source(source,frame,bg,condition,factory)
            row=dict(scene=spec['id'],motion=motion,frame=frame,condition=condition,K=len(ids),
                ids_bitwise=torch.equal(ids,actual_ids),radii_bitwise=bits_equal(radii,actual_radii),rgb_bitwise=bits_equal(reference,image))
            rows.append(row);assert all(row[x] for x in ['ids_bitwise','radii_bitwise','rgb_bitwise']),row
            retained.append((image,image.clone(),actual_ids,actual_ids.clone(),actual_radii,actual_radii.clone()))
        for a,b,c,d,e,f in retained:assert bits_equal(a,b) and torch.equal(c,d) and bits_equal(e,f),'输出被后续帧覆盖'
        retained.clear()
    ready=torch.cuda.Event();ready.record()
    for stream in [torch.cuda.Stream(),torch.cuda.Stream()]:
        with torch.cuda.stream(stream):
            stream.wait_event(ready);local=JointReleaseRenderer(source,student,mode,**{k:v for k,v in settings.items() if k!="members"})
            for condition in ['fixed0','fixed33','local0.1','local0.5']:
                frame=views[-1][1];cam=factory.from_source(source,frame)
                image,radii,ids,_=local.render_from_source(source,frame,bg,condition,factory)
                rows.append(dict(scene=spec['id'],condition=condition,stream=stream.cuda_stream,K=len(ids),finite=bool(torch.isfinite(image).all())))
                assert bool(torch.isfinite(image).all())
        stream.synchronize()
        try:local.render_from_source(source,views[0][1],bg,'local0.3',factory)
        except RuntimeError as error:assert 'per stream' in str(error) or 'per device/stream' in str(error)
        else:raise AssertionError('跨Stream复用没有拒绝')
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'checks.csv',index=False)
    save(out/'DONE.json',dict(passed=True,checks=len(rows),same_backend_bitwise=True,output_independent=True,stream_checked=True))
    print('STUDENT_JOINT_FINAL_EDGES_PASS',len(rows),flush=True)


if __name__=='__main__':main()
