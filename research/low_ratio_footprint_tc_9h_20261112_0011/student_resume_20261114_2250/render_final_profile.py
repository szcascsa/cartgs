"""冻结Student输出的代表性纯Rasterizer Profile；不与正式墙钟混用。"""
from student_study import *
from coherent_runtime import CoherentRenderer
from sign_runtime import SignRenderer
from encoded_runtime import EncodedRenderer
from encoded_sign_runtime import EncodedSignRenderer
from balanced_order import order


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ncu', action='store_true')
    parser.add_argument('--variant', default='old38')
    args=parser.parse_args()
    cases=[];owners=[]
    specs=student_specs(tune=True)[:1] if args.ncu else student_specs(tune=True)
    for spec in specs:
        source,student,_=load_case(spec);owners.append((source,student))
        network=network_variants(source,student)['fp32'][0]
        bg=source.scene.background
        runs={'old38':Renderer(38),'coherent155':CoherentRenderer(155),'sign166':SignRenderer(166),'encoded171':EncodedRenderer(171),'final182':EncodedSignRenderer(182),'F2':BaselineRenderer()}
        if args.ncu:runs={args.variant:runs[args.variant]}
        views=motions(source,quick=True)[::3]
        cam=source.camera(views[0][1])
        for engine in runs.values():engine.precompile(source.runtime.total,cam,bg)
        conditions=['local0.1'] if args.ncu else ['local0.01','local0.1','local0.3','local0.5','full']
        for condition in conditions:
            for _,frame in views[:1] if args.ncu else views:
                cam=source.camera(frame);packet,ids,_=p.request(network,cam,condition)
                for engine in runs.values():
                    for _ in range(3):engine.render(cam,packet,bg)
                cases.append((spec['slug'],condition,cam,packet,bg,runs))
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    for repeat in range(1 if args.ncu else 5):
        for j,(slug,condition,cam,packet,bg,runs) in enumerate(cases):
            for name in order(list(runs),repeat+j,reverse=repeat%2==1):
                torch.cuda.nvtx.range_push('/'.join(['nine_render',slug,'student_lod',condition,name]))
                runs[name].render(cam,packet,bg);torch.cuda.synchronize();torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    print('STUDENT_RASTER_PROFILE_COMPLETE',len(cases),flush=True)


if __name__=='__main__':main()
