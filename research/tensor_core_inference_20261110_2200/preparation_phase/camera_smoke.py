from prep_study import *
from native_camera import NativeCameraPreparer
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    for slug in TUNE:
        _,source=load_scene(slug);student,_,_=student_for(source,slug);run=make_engine(source,student,'C')
        old=create_camera_preparer(MiniCam,variant='R');new=NativeCameraPreparer(MiniCam)
        for frame in map(int,np.array([.07,.17,.43,.431,.57,.71,.93,.97])*(len(source.scene.views)-1)):
            a=old.from_source(source,frame);b=new.from_source(source,frame)
            row=dict(scene=slug,frame=frame)
            for key in ['world_view_transform','full_proj_transform','camera_center']:
                x=getattr(a,key);y=getattr(b,key);assert torch.isfinite(y).all()
                row[key+'_max_error']=float((x-y).abs().max());row[key+'_bitwise']=bits_equal(x,y)
            for ratio in [.1,.3,.5]:
                _,ia,sa=run.prepare(a,ratio_budget=ratio);_,ib,sb=run.prepare(b,ratio_budget=ratio)
                assert sa['candidate_gaussians']==sb['candidate_gaussians'] and len(ia)==len(ib),(slug,frame,sa,sb)
                overlap=int(torch.isin(ia,ib).sum());row[f'ratio{ratio}_same_ids']=torch.equal(ia,ib)
                row[f'ratio{ratio}_jaccard']=overlap/(len(ia)+len(ib)-overlap) if len(ia) else 1.
            rows.append(row)
        pose=np.asarray(source.scene.views[0]['pose']);saved=new.from_source(source,0)
        snapshot=[getattr(saved,k).clone() for k in ['world_view_transform','full_proj_transform','camera_center']]
        for frame in [1,50,2]:new.from_source(source,frame)
        assert all(bits_equal(x,getattr(saved,k)) for x,k in zip(snapshot,['world_view_transform','full_proj_transform','camera_center']))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):other=new.from_source(source,0)
        stream.synchronize();assert all(bits_equal(x,getattr(other,k)) for x,k in zip(snapshot,['world_view_transform','full_proj_transform','camera_center']))
        c=source.scene.calibration;fx=2*math.atan(c['width']/(2*c['fx']));fy=2*math.atan(c['height']/(2*c['fy']))
        for invalid in [np.zeros((4,4)),np.full((4,4),np.nan),np.full((4,4),np.inf)]:
            try:new.create_from_pose(c['width'],c['height'],fx,fy,invalid)
            except RuntimeError:pass
            else:raise AssertionError('Invalid camera accepted')
        write_csv(out/'checks.csv',rows);print('NATIVE_CAMERA_SMOKE',slug,len(rows),flush=True)
        del source,student,run,old,new;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,rows=len(rows),output_independence=True,nondefault_stream=True,invalid_rejected=True,
        binary_sha256=sha(dp.deployment_extension().__file__)))


if __name__=='__main__':main()
