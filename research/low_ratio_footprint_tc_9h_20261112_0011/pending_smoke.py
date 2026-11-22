"""异步联合调度的同权重节点验证；包含实际移动输入和finite GPU gate。"""
from joint_study import *
from pending_runtime import PendingRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--native',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    rows=[]
    for slug in TUNE:
        spec,source=load_source(slug);student,path,payload=student_for_spec(source,spec)
        for kind,s in [('teacher',None)]+([('student',student)] if student is not None else []):
            cmode=2 if s is not None else 0
            ref=p.make_engine(source,s,'D',frustum=1,projection=int(s is not None),device=1,compensator=cmode)
            new=PendingRenderer(source,s,mode,native_dispatch=args.native);bg=source.scene.background;factory=NativeCameraPreparer(p.MiniCam)
            frame0=motions(source,quick=True)[0][1];cam=factory.from_source(source,frame0)
            raster=Renderer(mode);raster.precompile(source.runtime.total,cam,bg);new.precompile(cam,bg)
            for _,frame in motions(source,quick=True):
                cam=factory.from_source(source,frame)
                for condition in ['fixed0','fixed1','local0.01','local0.1','local0.5','full']:
                    packet,ids,info=p.request(ref,cam,condition);baseline,radii=raster.render(cam,packet,bg)
                    image,r,ii,stats=new.render(cam,bg,condition)
                    row=dict(scene=spec['id'],model=kind,frame=frame,condition=condition,K=len(ids),
                        same_ids=torch.equal(ids,ii),radii_bitwise=bits_equal(r,radii),rgb_bitwise=bits_equal(image,baseline),
                        max_rgb_error=float((image-baseline).abs().max()))
                    rows.append(row);assert all(row[k] for k in ['same_ids','radii_bitwise','rgb_bitwise']),row
            # 异常前后均能恢复到相同当前帧，不吞finite错误，不让非法属性进入Raster。
            pipe=new.network._pipeline;cam=factory.from_source(source,frame0)
            call=(cam.world_view_transform,cam.full_proj_transform,cam.camera_center,cam.image_width,cam.image_height,float(cam.znear),0,.1,True,True,True)
            for status in [1,2]:
                pending=pipe._begin_pending(*call);flat,selected,control=pending
                control.view(torch.int32)[22]=status
                image,r=new.native.render_pending(bg,flat,control,len(selected),new.rest,cam.world_view_transform,
                    cam.full_proj_transform,math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),cam.image_height,cam.image_width,new.degree,cam.camera_center)
                assert bits_equal(image,bg[:,None,None].expand_as(image)),'非法帧未被GPU gate为背景'
                try:pipe._finish_pending()
                except RuntimeError as error:assert ('Selector' if status==1 else 'compensated') in str(error)
                else:raise AssertionError('finite exception dropped')
            pending=pipe._begin_pending(*call)
            try:pipe._begin_pending(*call)
            except RuntimeError as error:assert 'Finish pending' in str(error)
            else:raise AssertionError('duplicate pending accepted')
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                try:pipe._finish_pending()
                except RuntimeError as error:assert 'submission stream' in str(error)
                else:raise AssertionError('cross stream finish accepted')
            pipe._finish_pending()
            for value in [-.1,1.1,float('nan')]:
                invalid=call[:7]+(value,)+call[8:]
                try:pipe._begin_pending(*invalid)
                except RuntimeError:pass
                else:raise AssertionError('invalid ratio accepted')
            print('PENDING_SMOKE_MODEL_PASS',spec['id'],kind,flush=True)
            del ref,new,raster;gc.collect();torch.cuda.empty_cache()
        del source,student;gc.collect();torch.cuda.empty_cache()
    append(out/'checks.csv',rows);save(out/'SUMMARY.json',dict(passed=True,rows=len(rows),bitwise_vs_same_network=True,finite_gate=True,
        invalid_recovery=True,pending_exclusivity=True,stream_checks=True,no_images=True))


if __name__=='__main__':main()
