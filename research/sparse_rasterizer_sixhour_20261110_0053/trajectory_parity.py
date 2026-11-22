"""冻结候选的全轨迹数值审计；不记录测速结论，不保存渲染图片。"""
from harness import *
from final_benchmark import native_modules,CsvSink
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--conditions',default='full,local0.01,local0.05,local0.1');args=parser.parse_args()
    assert args.stride>=1
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    sink=CsvSink(output/'checks.csv');checks=0;scenes=[];started=time.time()
    for scene in SCENES:
        _,source=load_scene(scene['slug']);g=GraphRenderer(**config['graph']);c=SparseRenderer(**config['regular'])
        g.precompile(source.runtime.total,source.camera(0),source.scene.background);frames=len(source.scene.views)
        for frame in range(0,frames,args.stride):
            camera=source.camera(frame)
            for condition in args.conditions.split(','):
                packet,ids,stats=prepare(source,camera,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for name,engine in [('C',c),('G',g)]:
                    image,radii=engine.render(camera,packet,source.scene.background)
                    row=dict(scene=scene['id'],split=scene['split'],frame=frame,condition=condition,N=source.runtime.total,K=len(ids),variant=name,
                        rgb_bitwise=bits_equal(image,ref['render']),radii_bitwise=bits_equal(radii,ref['radii']))
                    sink.add(row);checks+=1
                    if not row['rgb_bitwise'] or not row['radii_bitwise']:
                        row.update(max_absolute_error=float((image-ref['render']).abs().max()),different_values=int((image.contiguous().view(torch.int32)!=ref['render'].contiguous().view(torch.int32)).sum()))
                        write_json(output/'FAILURE.json',row);sink.flush();raise AssertionError(row)
            if (frame+1)%100==0:
                sink.flush();write_json(output/'PROGRESS.json',dict(scene=scene['id'],frame=frame,frames=frames,checks=checks,elapsed_s=time.time()-started))
                print('TRAJECTORY_PARITY',scene['id'],frame+1,'/',frames,'checks',checks,flush=True)
        assert g.native.stats()[1:3]==[0,0]
        scenes.append(dict(scene=scene['id'],trajectory_frames=frames,checked_frames=len(range(0,frames,args.stride))))
        del source,g,c,packet,ref,image;torch.cuda.empty_cache();sink.flush()
    sink.close();write_json(output/'SUMMARY.json',dict(passed=True,scenes=scenes,comparisons=checks,stride=args.stride,
        conditions=args.conditions.split(','),elapsed_s=time.time()-started,images_saved=False,full_trajectory=args.stride==1))
    print('TRAJECTORY_PARITY_COMPLETE',checks,flush=True)


if __name__=='__main__':main()
