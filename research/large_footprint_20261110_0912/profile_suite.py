"""同一Nsight会话轮换G/C/D和实际运动输入，Profile耗时不代替正式性能。"""
from study_common import *
from footprint_runtime import Renderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());calls=0
    for slug in ['Replica__office2__rgbd']:
        _,source=load_scene(slug);views=views_for(source,'tune')[-4:]
        engines={'G':previous_renderer(),'C':Renderer(config['mode']),'D':Renderer(config['additional_modes']['D'])}
        for engine in engines.values():engine.precompile(source.runtime.total,source.camera(views[0][1]),source.scene.background)
        packets=[]
        for condition in ['local0.01','local0.05','local0.1','full']:
            for motion,frame in views:
                camera=source.camera(frame);packet,ids,_=prepare(source,camera,frame,condition)
                ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for engine in engines.values():
                    image,radii=engine.render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                packets.append((condition,camera,packet))
        for _,camera,packet in packets:
            for engine in engines.values():engine.render(camera,packet,source.scene.background)
        torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
        for repeat in range(3):
            for index,(condition,camera,packet) in enumerate(packets):
                for name in rotate(list(engines),index+repeat):
                    torch.cuda.nvtx.range_push(f'lfp/{slug}/{condition}/{name}')
                    image=engines[name].render(camera,packet,source.scene.background)[0]
                    torch.cuda.synchronize();torch.cuda.nvtx.range_pop();calls+=1
        torch.cuda.cudart().cudaProfilerStop()
        del source,engines,packets,packet,image,ref;torch.cuda.empty_cache()
    save_json(out/'SUMMARY.json',dict(passed=True,calls=calls,profiling_only=True,paired_same_inputs=True))
    print('FOOTPRINT_PROFILE_SUITE_COMPLETE',calls,flush=True)


if __name__=='__main__':main()
