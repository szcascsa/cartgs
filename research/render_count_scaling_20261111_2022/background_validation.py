"""复验生命周期与背景位模式；特别覆盖普通背景的空Tile快速判断。"""
from render_study import *
from background_runtime import Renderer,extension
import argparse
import lifecycle


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True)
    a.add_argument('--sanitizer',action='store_true');args=a.parse_args()
    stage=HERE/'background_stage'
    assert not (stage/'reports'/args.output).exists()
    # 原生命周期所有边界也必须通过；仅替换独立实验Renderer。
    lifecycle.Renderer=Renderer
    lifecycle.MODES={18:'background_guard'}
    lifecycle.HERE=stage
    saved=sys.argv[:]
    sys.argv=[sys.argv[0],'--output',args.output,'--modes','18']
    lifecycle.main()
    sys.argv=saved
    out=stage/'reports'/args.output
    checks=[]
    backgrounds=[[0.,0.,0.],[.13,.31,.76],[-.1,.2,.8],[-0.,0.,0.],[1.e-40,0.,0.],
                 [float('inf'),.1,0.],[float('nan'),.1,0.]]
    run=Renderer()
    cam=p.MiniCam(97,65,1.2,1.,np.eye(4))
    for background in backgrounds:
        bg=torch.tensor(background,device='cuda')
        for k in (0,1,2,13,31,32,33,256,257,513):
            for style in ('outside','no_alpha','barely_alpha','overlap','huge'):
                packet=synthetic_packet(k,style)
                ref=p.render_prepared(cam,packet,bg,return_radii=True)
                image,radii=run.render(cam,packet,bg)
                assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']),(background,k,style)
                checks.append(dict(background=str(background),K=k,style=style,bitwise=True))
    for slug in TUNE+['VECtor__corridors-dolly__stereo']:
        _,source=load_source(slug)
        frames=[653] if slug.startswith('VECtor__corridors') else [frame for _,frame in motions(source,quick=True)]
        for frame in frames:
            camera=source.camera(frame)
            for cond in ('local0.01','local0.05','local0.1','local0.3','local0.5','full'):
                packet,_,_=request(source,camera,cond)
                ref=p.render_prepared(camera,packet,source.scene.background,return_radii=True)
                image,radii=run.render(camera,packet,source.scene.background)
                assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']),(slug,frame,cond)
                checks.append(dict(background=slug,K=len(radii),style=cond,bitwise=True))
        del source,packet,ref
        gc.collect();torch.cuda.empty_cache()
    append(out/'background_checks.csv',checks)
    save(out/'BACKGROUND_PASS.json',dict(passed=True,checks=len(checks),binary_sha256=sha(extension().__file__)))
    print('BACKGROUND_VALIDATION_PASS',len(checks),flush=True)


if __name__=='__main__':main()
