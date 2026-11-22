"""验证实际Alpha参与诊断的数值口径，全部统计在计时外。"""
from harness import *
from runtime import SparseRenderer,extension


@torch.no_grad()
def diagnostics(engine,camera,packet,bg):
    out=engine.render(camera,packet,bg,True);P=len(out[1])
    if not P:
        tiles=math.ceil(camera.image_width/16)*math.ceil(camera.image_height/16)
        return dict(K=0,preprocess_visible=0,alpha_participating=0,alpha_interactions=0,tile_instances=0,bbox_instances=0,
            tiles=tiles,active_tiles=0,overflow_tiles=0,members_p50=0.,members_p95=0.,members_max=0,depth_words=0,
            nonempty_words_mean=0.,workspace_bytes=engine.native.workspace_bytes(),width=camera.image_width,height=camera.image_height)
    if len(out)>15 and int(out[15][0]):
        image=bg[:,None,None].expand(-1,camera.image_height,camera.image_width).clone()
        flags=out[1].new_zeros(P);count=out[1].new_zeros(1)
    else:image,flags,count=extension().diagnose_alpha(bg.contiguous(),out[2],out[3],out[13],out[9],out[7],out[14],camera.image_height,camera.image_width)
    assert bits_equal(image,out[0]),'Alpha诊断图必须与实际图逐位一致'
    counts=out[12].cpu().numpy().astype(np.int64);bits=out[9].cpu().numpy().astype(np.uint32)
    return dict(K=P,preprocess_visible=int((out[1]>0).sum()),alpha_participating=int(flags.sum()),
        alpha_interactions=int(count),tile_instances=int(counts.sum()),bbox_instances=int(out[5].sum()),
        tiles=len(counts),active_tiles=int((counts>0).sum()),overflow_tiles=int((counts>engine.capacity).sum()),
        members_p50=float(np.quantile(counts,.5)),members_p95=float(np.quantile(counts,.95)),members_max=int(counts.max()),
        depth_words=bits.shape[1],nonempty_words_mean=float((bits!=0).sum(axis=1).mean()),
        workspace_bytes=engine.native.workspace_bytes(),width=camera.image_width,height=camera.image_height)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[]
    engine=SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1)
    for scene in SCENES:
        if scene['split']!='tune':continue
        item,source=load_scene(scene['slug'])
        for motion,frame in views_for(source)[-2:]:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.1','full']:
                packet,ids,stats=prepare(source,camera,frame,condition)
                value=diagnostics(engine,camera,packet,source.scene.background)
                rows.append(dict(scene=scene['id'],condition=condition,frame=frame,**value))
        del source;torch.cuda.empty_cache();print('ALPHA_DIAGNOSTICS',scene['id'],flush=True)
    write_csv(output/'workloads.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,rows=rows,
        scope='原Alpha阈值、原透射率早停、仅统计实际写入颜色的Gaussian；诊断图逐位核对'))


if __name__=='__main__':main()
