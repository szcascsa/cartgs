"""冻结后继续定位像素并行瓶颈；只在原5个调优场景筛选新候选。"""
from harness import *
from final_benchmark import native_modules,rotate
from final_edges import packet as synthetic_packet
from runtime import SparseRenderer


def candidate_extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('CUDA_HOME','/usr/local/cuda-11.8');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.6')
    os.environ['MAX_JOBS']='2'
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:'+os.environ.get('PATH','')
    root=HERE/'late_candidates/microtile';build=root/'build';build.mkdir(exist_ok=True)
    return load(name='mycartgs_sparse_microtile_study_v1',sources=[str(root/'source'/name) for name in
        ['forward_sparse.cu','sparse_inference.cu','preprocess_sort.cu']],
        extra_include_paths=[str(root/'source/reference/cuda_rasterizer')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','-lineinfo'],build_directory=str(build),verbose=True)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    reference=SparseRenderer(**config['regular']);native=candidate_extension()
    variants={'C':reference}
    for name,variant in [('slice8',10),('slice4',11)]:
        engine=SparseRenderer.__new__(SparseRenderer);engine.__dict__.update(reference.__dict__)
        engine.native=native.Rasterizer();engine.render_variant=variant;variants[name]=engine
    checks=[];samples=[];events=[];torch.manual_seed(21701)
    def compare(label,camera,packet,bg):
        ref=render_prepared(camera,packet,bg,return_radii=True)
        for name,engine in variants.items():
            image,radii=engine.render(camera,packet,bg)
            row=dict(case=label,variant=name,K=len(radii),rgb_bitwise=bits_equal(image,ref['render']),radii_bitwise=bits_equal(radii,ref['radii']))
            checks.append(row)
            if not row['rgb_bitwise'] or not row['radii_bitwise']:
                row['max_abs']=float((image-ref['render']).abs().max());write_json(output/'FAILURE.json',row)
                write_csv(output/'correctness.csv',checks);raise AssertionError(row)
    for W,H in [(97,65),(641,479)]:
        camera=MiniCam(W,H,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
        for P in [0,1,31,255,256,257,511,513,1025]:
            for style in ['random','overlap','outside','no_alpha','huge','thin']:
                compare(f'{W}x{H}/{P}/{style}',camera,synthetic_packet(P,style),bg)
    for scene in SCENES:
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug']);views=views_for(source,'tune')[:2]+views_for(source,'tune')[-2:]
        for condition in ['local0.01','local0.05','local0.1','local0.7','full']:
            observation=external_load();assert not observation['active'],observation
            packets=[]
            for motion,frame in views:
                cam=source.camera(frame);p,ids,stats=prepare(source,cam,frame,condition)
                compare(scene['id']+'/'+condition+'/'+str(frame),cam,p,source.scene.background)
                packets.append((frame,cam,p))
            for repeat in range(6):
                for index in range(32):
                    frame,cam,p=packets[index%len(packets)]
                    for name in rotate(list(variants),index+repeat):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        image=variants[name].render(cam,p,source.scene.background)[0];torch.cuda.synchronize()
                        samples.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,
                            repeat=repeat,iteration=index,K=len(p['inputs']['means3D']),wall_ms=(time.perf_counter_ns()-tick)/1e6))
            for name,engine in variants.items():
                values=[]
                for iteration in range(20):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();image=engine.render(cam,p,source.scene.background)[0];end.record();end.synchronize();values.append(begin.elapsed_time(end))
                events.append(dict(scene=scene['id'],condition=condition,variant=name,event_ms=st.median(values)))
            write_csv(output/'samples.csv',samples);write_csv(output/'events.csv',events);write_csv(output/'correctness.csv',checks)
            print('MICROTILE_STUDY',scene['id'],condition,len(samples),flush=True)
        del source,packets,p;torch.cuda.empty_cache()
    write_json(output/'SUMMARY.json',dict(passed=True,checks=len(checks),samples=len(samples),tune_scenes_only=True,
        candidate_path=str(native.__file__),candidate_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        semantics='同一Tile成员和逐像素合成顺序；将16x16 Tile均匀拆为两个16x8或四个16x4 Block；所有K/Ratio走同一调度'))
    print('MICROTILE_STUDY_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
