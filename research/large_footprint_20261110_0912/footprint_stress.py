"""真实执行的合成压力输入；只解释footprint机制，不替代冻结真实场景验证。"""
from study_common import *
from footprint_runtime import Renderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);native_modules(BASELINE_CONFIG)
    frozen=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());rows=[];checks=[];torch.manual_seed(104729)
    for width,height in [(641,479),(1280,720)]:
        camera=MiniCam(width,height,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
        engines={'G':previous_renderer(),'C':Renderer(frozen['mode']),'D':Renderer(frozen['additional_modes']['D'])}
        for engine in engines.values():engine.precompile(512,camera,bg)
        functions={'A':lambda p:render_prepared(camera,p,bg)['render']}
        functions.update({n:(lambda p,e=e:e.render(camera,p,bg)[0]) for n,e in engines.items()})
        for points in [1,32,128,512]:
            for scale in [.03,.3,3.]:
                for opacity in [.02,.5]:
                    packet=synthetic_packet(points,'huge');packet['inputs']['scales'].fill_(scale)
                    packet['inputs']['opacities'].fill_(opacity)
                    ref=render_prepared(camera,packet,bg,return_radii=True)
                    for name,engine in engines.items():
                        image,radii=engine.render(camera,packet,bg)
                        assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                        snapshot=engine.native.snapshot() if hasattr(engine.native,'snapshot') else None
                        counts=snapshot[3].cpu().numpy().astype(np.int64) if snapshot is not None else None
                        checks.append(dict(width=width,height=height,K=points,scale=scale,opacity=opacity,variant=name,
                            RGB_bitwise=True,radii_bitwise=True,tile_instances=-1 if counts is None else int(counts.sum()),
                            members_max=-1 if counts is None else int(counts.max()),
                            overflow_tiles=-1 if counts is None else int((counts>256).sum()),preprocess_visible=int((radii>0).sum())))
                    for repeat in range(20):
                        for name in rotate(list(functions),repeat+points):
                            torch.cuda.synchronize();tick=time.perf_counter_ns();image=functions[name](packet)
                            torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-tick)/1e6
                            rows.append(dict(width=width,height=height,K=points,scale=scale,opacity=opacity,
                                repeat=repeat,variant=name,wall_ms=elapsed))
                    write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks)
            print('FOOTPRINT_STRESS',width,height,points,flush=True)
        del engines,functions,packet,image,ref,snapshot;torch.cuda.empty_cache()
    save_json(out/'SUMMARY.json',dict(passed=True,comparisons=len(checks),samples=len(rows),
        synthetic_inputs=True,actual_CUDA_measurements=True,scope='合成压力输入，不能冒充真实场景通用性结果'))
    print('FOOTPRINT_STRESS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
